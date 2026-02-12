"""
Database repository for CRUD operations.

Provides a high-level interface for database operations
on all models (user, turn, conversation, inference, memory).
"""

import json
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, User, Turn, Inference, Conversation, Memory, MemoryEdge


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
        
        # Apply schema migrations for existing databases
        self._apply_migrations()
    
    def _apply_migrations(self):
        """Apply schema migrations for columns added after initial release.
        
        Migrations handled:
        1. Inference table: Add task, conversation_id columns
        2. Memory table: Add embedding column (BLOB for semantic retrieval)
        3. memory_edge table: Create if missing (for graph-based retrieval)
        """
        from sqlalchemy import inspect, text
        inspector = inspect(self.engine)
        
        table_names = inspector.get_table_names()
        
        # --- Inference table migration ---
        if "inference" in table_names:
            columns = {col["name"] for col in inspector.get_columns("inference")}
            needs_full_migration = "task" not in columns
            needs_conversation_id = "task" in columns and "conversation_id" not in columns
            
            if needs_full_migration or needs_conversation_id:
                with self.engine.connect() as conn:
                    conn.execute(text("PRAGMA foreign_keys = OFF"))
                    conn.execute(text("""
                        CREATE TABLE inference_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            turn_id INTEGER REFERENCES turn(id),
                            conversation_id INTEGER REFERENCES conversation(id),
                            task VARCHAR(50) DEFAULT 'conversation',
                            prompt TEXT NOT NULL,
                            response TEXT NOT NULL,
                            model VARCHAR(100) NOT NULL,
                            provider VARCHAR(50) NOT NULL,
                            prompt_tokens INTEGER NOT NULL,
                            completion_tokens INTEGER NOT NULL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(turn_id, task)
                        )
                    """))
                    
                    if needs_full_migration:
                        conn.execute(text("""
                            INSERT INTO inference_new
                                (id, turn_id, conversation_id, task, prompt, response,
                                 model, provider, prompt_tokens, completion_tokens, timestamp)
                            SELECT i.id, i.turn_id, t.conversation_id, 'conversation',
                                   i.prompt, i.response, i.model, i.provider,
                                   i.prompt_tokens, i.completion_tokens, i.timestamp
                            FROM inference i
                            LEFT JOIN turn t ON t.id = i.turn_id
                        """))
                    else:
                        conn.execute(text("""
                            INSERT INTO inference_new
                                (id, turn_id, conversation_id, task, prompt, response,
                                 model, provider, prompt_tokens, completion_tokens, timestamp)
                            SELECT i.id, i.turn_id, t.conversation_id, i.task,
                                   i.prompt, i.response, i.model, i.provider,
                                   i.prompt_tokens, i.completion_tokens, i.timestamp
                            FROM inference i
                            LEFT JOIN turn t ON t.id = i.turn_id
                        """))
                    
                    conn.execute(text("DROP TABLE inference"))
                    conn.execute(text("ALTER TABLE inference_new RENAME TO inference"))
                    conn.execute(text("PRAGMA foreign_keys = ON"))
                    conn.commit()
        
        # --- Memory table: add embedding column ---
        if "memory" in table_names:
            memory_columns = {col["name"] for col in inspector.get_columns("memory")}
            if "embedding" not in memory_columns:
                with self.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE memory ADD COLUMN embedding BLOB"))
                    conn.commit()
        
        # --- memory_edge table: create if missing ---
        if "memory_edge" not in table_names:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE memory_edge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_memory_id INTEGER NOT NULL REFERENCES memory(id),
                        target_memory_id INTEGER NOT NULL REFERENCES memory(id),
                        label VARCHAR(100) NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """))
                conn.commit()
    
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
    
    # Conversation operations
    
    def create_conversation(self, user_id: int) -> Conversation:
        """
        Create a new conversation for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Created Conversation instance
        """
        with self.get_session() as session:
            conversation = Conversation(user_id=user_id)
            session.add(conversation)
            session.flush()
            session.refresh(conversation)
            session.expunge(conversation)
            return conversation
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation instance or None
        """
        with self.get_session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation:
                session.expunge(conversation)
            return conversation
    
    def get_active_conversation(self, user_id: int) -> Optional[Conversation]:
        """
        Get the active conversation for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Active Conversation instance or None
        """
        with self.get_session() as session:
            conversation = session.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.is_active == True
            ).first()
            if conversation:
                session.expunge(conversation)
            return conversation
    
    def end_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Mark a conversation as ended.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Updated Conversation instance or None
        """
        with self.get_session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation:
                conversation.is_active = False
                conversation.ended_at = datetime.utcnow()
                session.flush()
                session.refresh(conversation)
                session.expunge(conversation)
            return conversation
    
    def get_conversation_turns(self, conversation_id: int) -> List[Turn]:
        """
        Get all turns for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of Turn instances ordered by turn_number
        """
        with self.get_session() as session:
            turns = session.query(Turn).filter(
                Turn.conversation_id == conversation_id
            ).order_by(Turn.turn_number).all()
            
            for turn in turns:
                session.expunge(turn)
            return turns
    
    # Turn operations
    
    def add_turn(
        self,
        user_id: int,
        role: str,
        content: str,
        turn_number: int,
        conversation_id: Optional[int] = None
    ) -> Turn:
        """
        Add a turn for a user.
        
        Args:
            user_id: User ID
            role: Role (user/assistant/system)
            content: Message content
            turn_number: Turn number for this user
            conversation_id: Optional conversation ID
            
        Returns:
            Created Turn instance
        """
        with self.get_session() as session:
            turn = Turn(
                user_id=user_id,
                role=role,
                content=content,
                turn_number=turn_number,
                conversation_id=conversation_id
            )
            session.add(turn)
            session.flush()
            session.refresh(turn)
            session.expunge(turn)
            return turn
    
    def get_user_turns(
        self,
        user_id: int,
        limit: Optional[int] = None,
        exclude_conversation_id: Optional[int] = None
    ) -> List[Turn]:
        """
        Get turns for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of turns to return (most recent)
            exclude_conversation_id: Conversation ID to exclude from results
            
        Returns:
            List of Turn instances ordered by turn_number
        """
        with self.get_session() as session:
            query = session.query(Turn).filter(
                Turn.user_id == user_id
            )
            
            if exclude_conversation_id is not None:
                query = query.filter(Turn.conversation_id != exclude_conversation_id)
            
            query = query.order_by(Turn.turn_number.desc())
            
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
        prompt: str,
        response: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        turn_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        task: str = "conversation"
    ) -> Inference:
        """
        Add an inference log.
        
        Args:
            prompt: Full prompt sent to LLM
            response: LLM response
            model: Model name
            provider: LLM provider
            prompt_tokens: Input token count
            completion_tokens: Output token count
            turn_id: Associated turn ID (set for conversation tasks)
            conversation_id: Associated conversation/session ID (set for all tasks)
            task: Task identifier (conversation, memory_extraction, memory_consolidation)
            
        Returns:
            Created Inference instance
        """
        with self.get_session() as session:
            inference = Inference(
                turn_id=turn_id,
                conversation_id=conversation_id,
                task=task,
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
    
    # Memory operations
    
    def add_memory(
        self,
        conversation_id: int,
        statement: str,
        memory_type: str,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """
        Add a memory for a conversation.
        
        Args:
            conversation_id: Conversation ID
            statement: Concise memory statement
            memory_type: Type of memory (semantic, episodic, procedural, affective)
            tags: Optional list of tags for the memory
            
        Returns:
            Created Memory instance (is_active=True by default)
        """
        with self.get_session() as session:
            memory = Memory(
                conversation_id=conversation_id,
                statement=statement,
                memory_type=memory_type,
                is_active=True
            )
            if tags:
                memory.set_tags(tags)
            session.add(memory)
            session.flush()
            session.refresh(memory)
            session.expunge(memory)
            return memory
    
    def add_memories_batch(self, conversation_id: int, memories: List[dict]) -> List[Memory]:
        """
        Add multiple memories for a conversation.
        
        Args:
            conversation_id: Conversation ID
            memories: List of dicts with statement, type, and optional tags keys
            
        Returns:
            List of created Memory instances (all is_active=True)
        """
        with self.get_session() as session:
            created_memories = []
            for mem_data in memories:
                memory = Memory(
                    conversation_id=conversation_id,
                    statement=mem_data["statement"],
                    memory_type=mem_data["type"],
                    is_active=True
                )
                tags = mem_data.get("tags")
                if tags:
                    memory.set_tags(tags)
                session.add(memory)
                created_memories.append(memory)
            
            session.flush()
            for memory in created_memories:
                session.refresh(memory)
                session.expunge(memory)
            
            return created_memories
    
    def get_memories(self, conversation_id: int) -> List[Memory]:
        """
        Get memories for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of Memory instances
        """
        with self.get_session() as session:
            memories = session.query(Memory).filter(
                Memory.conversation_id == conversation_id
            ).all()
            
            for memory in memories:
                session.expunge(memory)
            return memories
    
    def get_user_memories(self, user_id: int, limit: Optional[int] = None) -> List[Memory]:
        """
        Get all memories for a user across all conversations (including inactive).
        
        Args:
            user_id: User ID
            limit: Maximum number of memories to return
            
        Returns:
            List of Memory instances ordered by created_at desc
        """
        with self.get_session() as session:
            query = session.query(Memory).join(
                Conversation
            ).filter(
                Conversation.user_id == user_id
            ).order_by(Memory.created_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            memories = query.all()
            
            for memory in memories:
                session.expunge(memory)
            return memories
    
    def get_user_active_memories(self, user_id: int, limit: Optional[int] = None) -> List[Memory]:
        """
        Get all active memories for a user across all conversations.
        
        Args:
            user_id: User ID
            limit: Maximum number of memories to return
            
        Returns:
            List of active Memory instances ordered by created_at desc
        """
        with self.get_session() as session:
            query = (
                session.query(Memory)
                .join(Conversation)
                .filter(Conversation.user_id == user_id, Memory.is_active == True)
                .order_by(Memory.created_at.desc())
            )
            
            if limit:
                query = query.limit(limit)
            
            memories = query.all()
            
            for memory in memories:
                session.expunge(memory)
            return memories
    
    def get_user_active_memories_by_tags(self, user_id: int, tags: List[str]) -> List[Memory]:
        """
        Get active memories for a user where at least one tag matches.
        
        Uses Python-side filtering since tags are stored as JSON text
        in SQLite (no native JSON array operations).
        
        Args:
            user_id: User ID
            tags: List of tags to match against
            
        Returns:
            List of active Memory instances that share at least one tag
        """
        all_active = self.get_user_active_memories(user_id, limit=None)
        if not tags:
            return all_active
        
        target_tags = {t.lower().strip() for t in tags if t.strip()}
        matched = []
        for memory in all_active:
            memory_tags = set(memory.get_tags())
            if memory_tags & target_tags:
                matched.append(memory)
        return matched
    
    def update_memory_content(self, memory_id: int, statement: str, tags: Optional[List[str]] = None) -> Optional[Memory]:
        """
        Update an existing memory's statement and/or tags (MERGE operation).
        
        Sets updated_at to current time. Keeps is_active=True.
        
        Args:
            memory_id: ID of the memory to update
            statement: New statement text
            tags: Optional new tags list (merged with existing if provided)
            
        Returns:
            Updated Memory instance or None if not found
        """
        with self.get_session() as session:
            memory = session.get(Memory, memory_id)
            if not memory:
                return None
            
            memory.statement = statement
            memory.updated_at = datetime.utcnow()
            
            if tags is not None:
                # Merge new tags with existing
                existing_tags = memory.get_tags()
                merged = list(set(existing_tags + tags))
                memory.set_tags(merged)
            
            session.flush()
            session.refresh(memory)
            session.expunge(memory)
            return memory
    
    def mark_memory_superseded(self, memory_id: int, superseded_by_id: int) -> Optional[Memory]:
        """
        Mark a memory as superseded by a newer memory (UPDATE operation).
        
        Sets is_active=False, superseded_at=now(), superseded_by=new_id.
        The memory record is retained for audit trail.
        
        Args:
            memory_id: ID of the memory to mark as superseded
            superseded_by_id: ID of the newer memory that replaces this one
            
        Returns:
            Updated Memory instance or None if not found
        """
        with self.get_session() as session:
            memory = session.get(Memory, memory_id)
            if not memory:
                return None
            
            memory.is_active = False
            memory.superseded_at = datetime.utcnow()
            memory.superseded_by = superseded_by_id
            
            session.flush()
            session.refresh(memory)
            session.expunge(memory)
            return memory
    
    def get_memory_history(self, memory_id: int) -> List[Memory]:
        """
        Retrieve the chain of superseded memories leading to the given memory.
        
        Follows the superseded_by chain backwards for audit trail.
        
        Args:
            memory_id: ID of the current active memory
            
        Returns:
            List of Memory instances from oldest to newest
        """
        with self.get_session() as session:
            # Find all memories that were superseded leading to this one
            chain = []
            predecessors = session.query(Memory).filter(
                Memory.superseded_by == memory_id
            ).all()
            
            for pred in predecessors:
                session.expunge(pred)
                # Recursively get older predecessors
                older = self.get_memory_history(pred.id)
                chain.extend(older)
                chain.append(pred)
            
            return chain
    
    # Aggregate / stats operations
    
    def count_user_conversations(self, user_id: int) -> int:
        """
        Count total conversations (sessions) for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of conversations
        """
        with self.get_session() as session:
            return session.query(Conversation).filter(
                Conversation.user_id == user_id
            ).count()
    
    def count_user_active_memories(self, user_id: int) -> int:
        """
        Count active memories for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of active memories
        """
        with self.get_session() as session:
            return session.query(Memory).join(Conversation).filter(
                Conversation.user_id == user_id,
                Memory.is_active == True
            ).count()
    
    # Embedding operations
    
    def update_memory_embedding(self, memory_id: int, embedding_bytes: bytes) -> Optional[Memory]:
        """
        Store an embedding vector for a memory.
        
        Args:
            memory_id: ID of the memory
            embedding_bytes: Serialized numpy array (via np.ndarray.tobytes())
            
        Returns:
            Updated Memory instance or None if not found
        """
        with self.get_session() as session:
            memory = session.get(Memory, memory_id)
            if not memory:
                return None
            memory.embedding = embedding_bytes
            session.flush()
            session.refresh(memory)
            session.expunge(memory)
            return memory
    
    def get_user_active_memories_with_embeddings(self, user_id: int) -> List[Memory]:
        """
        Get all active memories for a user, including those with and without
        embeddings. Callers should check memory.embedding is not None.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active Memory instances ordered by created_at desc
        """
        return self.get_user_active_memories(user_id, limit=None)
    
    # MemoryEdge operations
    
    def add_memory_edge(self, source_id: int, target_id: int, label: str) -> MemoryEdge:
        """
        Create an edge between two memories in the graph.
        
        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            label: Relationship label (e.g., 'HEALTH_LIFESTYLE')
            
        Returns:
            Created MemoryEdge instance
        """
        with self.get_session() as session:
            edge = MemoryEdge(
                source_memory_id=source_id,
                target_memory_id=target_id,
                label=label,
                is_active=True
            )
            session.add(edge)
            session.flush()
            session.refresh(edge)
            session.expunge(edge)
            return edge
    
    def get_edges_for_memories(self, memory_ids: List[int]) -> List[MemoryEdge]:
        """
        Get all active edges where source or target is in the given set.
        
        Args:
            memory_ids: List of memory IDs to find edges for
            
        Returns:
            List of active MemoryEdge instances
        """
        if not memory_ids:
            return []
        
        with self.get_session() as session:
            from sqlalchemy import or_
            edges = session.query(MemoryEdge).filter(
                MemoryEdge.is_active == True,
                or_(
                    MemoryEdge.source_memory_id.in_(memory_ids),
                    MemoryEdge.target_memory_id.in_(memory_ids)
                )
            ).all()
            
            for edge in edges:
                session.expunge(edge)
            return edges
    
    def deactivate_edges_for_memory(self, memory_id: int) -> int:
        """
        Mark all active edges involving a memory as inactive.
        
        Called when a memory is superseded or needs edge re-evaluation.
        
        Args:
            memory_id: ID of the memory whose edges should be deactivated
            
        Returns:
            Number of edges deactivated
        """
        with self.get_session() as session:
            from sqlalchemy import or_
            count = session.query(MemoryEdge).filter(
                MemoryEdge.is_active == True,
                or_(
                    MemoryEdge.source_memory_id == memory_id,
                    MemoryEdge.target_memory_id == memory_id
                )
            ).update({"is_active": False}, synchronize_session="fetch")
            return count
    
    def get_connected_memories(self, memory_ids: List[int], limit: Optional[int] = None) -> List[Memory]:
        """
        1-hop graph traversal: get active memories connected to the given set
        via active edges.
        
        Returns memories that are NOT already in the seed set.
        
        Args:
            memory_ids: Seed memory IDs to traverse from
            limit: Optional maximum number of connected memories to return
            
        Returns:
            List of connected Memory instances (excluding seeds)
        """
        if not memory_ids:
            return []
        
        edges = self.get_edges_for_memories(memory_ids)
        if not edges:
            return []
        
        # Collect IDs on the other side of each edge
        seed_set = set(memory_ids)
        connected_ids = set()
        for edge in edges:
            if edge.source_memory_id in seed_set:
                connected_ids.add(edge.target_memory_id)
            if edge.target_memory_id in seed_set:
                connected_ids.add(edge.source_memory_id)
        
        # Remove seeds from connected set
        connected_ids -= seed_set
        
        if not connected_ids:
            return []
        
        with self.get_session() as session:
            query = session.query(Memory).filter(
                Memory.id.in_(connected_ids),
                Memory.is_active == True
            )
            if limit:
                query = query.limit(limit)
            
            memories = query.all()
            for memory in memories:
                session.expunge(memory)
            return memories
    
    def get_connected_memories_with_edges(
        self, memory_ids: List[int]
    ) -> List[tuple]:
        """
        1-hop graph traversal returning (Memory, edge_label) tuples.
        
        Useful for the graph retriever to prioritize by edge label.
        
        Args:
            memory_ids: Seed memory IDs to traverse from
            
        Returns:
            List of (Memory, edge_label) tuples for connected memories
        """
        if not memory_ids:
            return []
        
        edges = self.get_edges_for_memories(memory_ids)
        if not edges:
            return []
        
        seed_set = set(memory_ids)
        # Map connected_id -> list of edge labels
        connected_labels: dict = {}
        for edge in edges:
            if edge.source_memory_id in seed_set:
                other_id = edge.target_memory_id
            elif edge.target_memory_id in seed_set:
                other_id = edge.source_memory_id
            else:
                continue
            if other_id not in seed_set:
                connected_labels.setdefault(other_id, []).append(edge.label)
        
        if not connected_labels:
            return []
        
        with self.get_session() as session:
            memories = session.query(Memory).filter(
                Memory.id.in_(connected_labels.keys()),
                Memory.is_active == True
            ).all()
            
            results = []
            for memory in memories:
                session.expunge(memory)
                labels = connected_labels.get(memory.id, [])
                for label in labels:
                    results.append((memory, label))
            return results
