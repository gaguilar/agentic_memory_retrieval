"""
SQLAlchemy ORM models for the conversational agent system.

Defines tables for:
- user: User accounts
- conversation: Conversation sessions
- turn: Individual messages from users
- inference: LLM API call logs
- memory: Memories extracted from conversations
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Enum as SQLEnum,
    UniqueConstraint, LargeBinary
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
import enum

class TurnRole(str, enum.Enum):
    """Role of a turn in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryType(str, enum.Enum):
    """Type of extracted memory."""
    SEMANTIC = "semantic"      # Facts, knowledge, general information
    EPISODIC = "episodic"      # Events, experiences, specific occurrences
    PROCEDURAL = "procedural"  # How-to, processes, methods
    AFFECTIVE = "affective"    # Emotions, preferences, feelings


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class User(Base):
    """User model representing a chat participant."""
    __tablename__ = "user"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    turns: Mapped[List["Turn"]] = relationship(
        "Turn", back_populates="user", cascade="all, delete-orphan",
        order_by="Turn.turn_number"
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan",
        order_by="Conversation.started_at"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class Conversation(Base):
    """A conversation session grouping multiple turns."""
    __tablename__ = "conversation"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    turns: Mapped[List["Turn"]] = relationship("Turn", back_populates="conversation", cascade="all, delete-orphan", order_by="Turn.turn_number")
    memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="conversation", cascade="all, delete-orphan")
    inferences: Mapped[List["Inference"]] = relationship("Inference", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        status = "active" if self.is_active else "ended"
        return f"<Conversation(id={self.id}, user_id={self.user_id}, status='{status}')>"


class Turn(Base):
    """Individual message turn from a user."""
    __tablename__ = "turn"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.id"), nullable=False)
    conversation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("conversation.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="turns")
    conversation: Mapped[Optional["Conversation"]] = relationship("Conversation", back_populates="turns")
    inference: Mapped[Optional["Inference"]] = relationship("Inference", back_populates="turn", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Turn(id={self.id}, role='{self.role}', turn={self.turn_number})>"


class Inference(Base):
    """Log of LLM API calls including prompts and responses.
    
    Each inference is tied to a `task` that identifies its purpose:
      - "conversation"         — normal chat turn (turn_id is set)
      - "memory_extraction"    — extracting memories from a session
      - "memory_consolidation" — consolidating a single memory candidate
    
    Source references:
      - `turn_id`         — FK to turn; set for conversation tasks
      - `conversation_id` — FK to conversation (session); set for all tasks
    
    Uniqueness:
      The pair (turn_id, task) is unique so that each turn has at most one
      inference per task type. Memory tasks may produce multiple inferences
      per conversation (e.g. one consolidation call per candidate).
    """
    __tablename__ = "inference"
    __table_args__ = (
        UniqueConstraint("turn_id", "task", name="uq_inference_turn_task"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("turn.id"), nullable=True
    )
    conversation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("conversation.id"), nullable=True)
    task: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="conversation")
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    turn: Mapped[Optional["Turn"]] = relationship("Turn", back_populates="inference")
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation", back_populates="inferences"
    )
    
    def __repr__(self) -> str:
        task_str = self.task or "unknown"
        return f"<Inference(id={self.id}, task='{task_str}', model='{self.model}', tokens={self.prompt_tokens}+{self.completion_tokens})>"


class Memory(Base):
    """Memory extracted from a completed conversation.
    
    Memories are never deleted. When superseded by newer information,
    they are marked as inactive (is_active=False) with a reference
    to the replacing memory (superseded_by).
    """
    __tablename__ = "memory"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversation.id"), nullable=False
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Tag-based indexing (JSON-encoded list of lowercase tags)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Embedding for semantic retrieval (serialized numpy array)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    
    # Soft-delete / versioning fields
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    superseded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    superseded_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("memory.id"), nullable=True
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="memories"
    )
    
    def get_tags(self) -> List[str]:
        """Parse the JSON tags column into a Python list."""
        if not self.tags:
            return []
        try:
            import json
            parsed = json.loads(self.tags)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_tags(self, tags: List[str]) -> None:
        """Serialize a Python list of tags into the JSON tags column."""
        import json
        self.tags = json.dumps(sorted(set(t.lower().strip() for t in tags if t.strip())))
    
    def __repr__(self) -> str:
        status = "active" if self.is_active else "superseded"
        return f"<Memory(id={self.id}, type='{self.memory_type}', status='{status}')>"


class MemoryEdge(Base):
    """An edge in the memory-to-memory graph.
    
    Represents a meaningful relationship between two memories.
    Edges are directed (source → target) with a descriptive label
    indicating the nature of the relationship.
    
    When a memory is superseded or merged, its edges are deactivated
    (is_active=False) and re-evaluated for the replacement memory.
    """
    __tablename__ = "memory_edge"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_memory_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("memory.id"), nullable=False
    )
    target_memory_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("memory.id"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    source_memory: Mapped["Memory"] = relationship(
        "Memory", foreign_keys=[source_memory_id]
    )
    target_memory: Mapped["Memory"] = relationship(
        "Memory", foreign_keys=[target_memory_id]
    )
    
    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return (
            f"<MemoryEdge(id={self.id}, "
            f"{self.source_memory_id}--[{self.label}]-->{self.target_memory_id}, "
            f"status='{status}')>"
        )
