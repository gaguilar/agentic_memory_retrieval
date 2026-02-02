"""
SQLAlchemy ORM models for the conversational agent system.

Defines tables for:
- user: User accounts
- turn: Individual messages from users
- inference: LLM API call logs
- prompt_memories: RAG memories included in prompts
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
import enum


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class TurnRole(str, enum.Enum):
    """Role of a turn in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


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
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class Turn(Base):
    """Individual message turn from a user."""
    __tablename__ = "turn"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="turns")
    inference: Mapped[Optional["Inference"]] = relationship(
        "Inference", back_populates="turn", uselist=False, cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Turn(id={self.id}, role='{self.role}', turn={self.turn_number})>"


class Inference(Base):
    """Log of LLM API calls including prompts and responses."""
    __tablename__ = "inference"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("turn.id"), nullable=False, unique=True
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    turn: Mapped["Turn"] = relationship("Turn", back_populates="inference")
    prompt_memories: Mapped[Optional["PromptMemories"]] = relationship(
        "PromptMemories", back_populates="inference", uselist=False,
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Inference(id={self.id}, model='{self.model}', tokens={self.prompt_tokens}+{self.completion_tokens})>"


class PromptMemories(Base):
    """Memories that were retrieved and included in the prompt context."""
    __tablename__ = "prompt_memories"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inference_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inference.id"), nullable=False, unique=True
    )
    formatted_memories: Mapped[str] = mapped_column(Text, nullable=False)
    retriever_type: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    inference: Mapped["Inference"] = relationship(
        "Inference", back_populates="prompt_memories"
    )
    
    def __repr__(self) -> str:
        return f"<PromptMemories(id={self.id}, retriever='{self.retriever_type}')>"
