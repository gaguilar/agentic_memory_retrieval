"""
Database repository for CRUD operations.

Provides a high-level interface for database operations
on all models (user, turn, inference, prompt_memories).
"""

from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, User, Turn, Inference, PromptMemories


class Repository:
    """
    Repository class for database operations.
    
    Provides CRUD operations for all models and handles
    database session management.
    """
    
    def __init__(self, database_url: str = "sqlite:///conversations.db"):
        """
        Initialize the repository.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(self.engine)
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.
        
        Yields:
            Database session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # User operations
    
    def create_user(self, username: str) -> User:
        """
        Create a new user.
        
        Args:
            username: Unique username
            
        Returns:
            Created User instance
        """
        with self.get_session() as session:
            user = User(username=username)
            session.add(user)
            session.flush()
            session.refresh(user)
            # Expunge to detach from session
            session.expunge(user)
            return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User instance or None
        """
        with self.get_session() as session:
            user = session.get(User, user_id)
            if user:
                session.expunge(user)
            return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by username.
        
        Args:
            username: Username to find
            
        Returns:
            User instance or None
        """
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if user:
                session.expunge(user)
            return user
    
    def get_or_create_user(self, username: str) -> User:
        """
        Get existing user or create new one.
        
        Args:
            username: Username
            
        Returns:
            User instance
        """
        user = self.get_user_by_username(username)
        if user:
            return user
        return self.create_user(username)
    
    # Turn operations
    
    def add_turn(
        self,
        user_id: int,
        role: str,
        content: str,
        turn_number: int
    ) -> Turn:
        """
        Add a turn for a user.
        
        Args:
            user_id: User ID
            role: Role (user/assistant/system)
            content: Message content
            turn_number: Turn number for this user
            
        Returns:
            Created Turn instance
        """
        with self.get_session() as session:
            turn = Turn(
                user_id=user_id,
                role=role,
                content=content,
                turn_number=turn_number
            )
            session.add(turn)
            session.flush()
            session.refresh(turn)
            session.expunge(turn)
            return turn
    
    def get_user_turns(
        self,
        user_id: int,
        limit: Optional[int] = None
    ) -> List[Turn]:
        """
        Get turns for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of turns to return (most recent)
            
        Returns:
            List of Turn instances ordered by turn_number
        """
        with self.get_session() as session:
            query = session.query(Turn).filter(
                Turn.user_id == user_id
            ).order_by(Turn.turn_number.desc())
            
            if limit:
                query = query.limit(limit)
            
            turns = query.all()
            # Reverse to get chronological order
            turns = list(reversed(turns))
            
            for turn in turns:
                session.expunge(turn)
            return turns
    
    def get_last_turn_number(self, user_id: int) -> int:
        """
        Get the last turn number for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Last turn number or 0 if no turns
        """
        with self.get_session() as session:
            last_turn = session.query(Turn).filter(
                Turn.user_id == user_id
            ).order_by(Turn.turn_number.desc()).first()
            
            return last_turn.turn_number if last_turn else 0
    
    # Inference operations
    
    def add_inference(
        self,
        turn_id: int,
        prompt: str,
        response: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Inference:
        """
        Add an inference log for a turn.
        
        Args:
            turn_id: Associated turn ID
            prompt: Full prompt sent to LLM
            response: LLM response
            model: Model name
            provider: LLM provider
            prompt_tokens: Input token count
            completion_tokens: Output token count
            
        Returns:
            Created Inference instance
        """
        with self.get_session() as session:
            inference = Inference(
                turn_id=turn_id,
                prompt=prompt,
                response=response,
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            session.add(inference)
            session.flush()
            session.refresh(inference)
            session.expunge(inference)
            return inference
    
    def get_inference(self, inference_id: int) -> Optional[Inference]:
        """
        Get an inference by ID.
        
        Args:
            inference_id: Inference ID
            
        Returns:
            Inference instance or None
        """
        with self.get_session() as session:
            inference = session.get(Inference, inference_id)
            if inference:
                session.expunge(inference)
            return inference
    
    # Prompt memories operations
    
    def add_prompt_memories(
        self,
        inference_id: int,
        formatted_memories: str,
        retriever_type: str
    ) -> PromptMemories:
        """
        Add prompt memories record.
        
        Args:
            inference_id: Associated inference ID
            formatted_memories: Formatted memories text
            retriever_type: Type of retriever used
            
        Returns:
            Created PromptMemories instance
        """
        with self.get_session() as session:
            memories = PromptMemories(
                inference_id=inference_id,
                formatted_memories=formatted_memories,
                retriever_type=retriever_type
            )
            session.add(memories)
            session.flush()
            session.refresh(memories)
            session.expunge(memories)
            return memories
    
    def get_prompt_memories(
        self,
        memories_id: int
    ) -> Optional[PromptMemories]:
        """
        Get prompt memories by ID.
        
        Args:
            memories_id: PromptMemories ID
            
        Returns:
            PromptMemories instance or None
        """
        with self.get_session() as session:
            memories = session.get(PromptMemories, memories_id)
            if memories:
                session.expunge(memories)
            return memories
