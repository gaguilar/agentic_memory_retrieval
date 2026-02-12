"""
Tag embedding and similarity computation for memory consolidation.

Uses sentence-transformers/all-MiniLM-L6-v2 to embed tags and compute
cosine similarity for fuzzy tag matching when exact matches fail.
"""

import logging
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import transformers
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class TagEmbedder:
    """
    Embeds tags using a lightweight sentence-transformer model and
    computes cosine similarity for fuzzy tag matching.
    
    The model is loaded lazily on first use to avoid startup overhead.
    Tag embeddings are cached in-memory for efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the tag embedder.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
            similarity_threshold: Minimum cosine similarity to consider tags as matching
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}
    
    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = self._load_model_quiet()
            logger.info("Embedding model loaded successfully")
        return self._model

    def _load_model_quiet(self) -> "SentenceTransformer":
        """Load the model while suppressing noisy warnings, logs, and progress bars."""
        # Temporarily silence noisy loggers during model load
        noisy_loggers = [
            "sentence_transformers",
            "transformers",
            "huggingface_hub",
            "httpx",
            "urllib3",
            "mlx",
        ]
        saved_levels = {}
        for name in noisy_loggers:
            lg = logging.getLogger(name)
            saved_levels[name] = lg.level
            lg.setLevel(logging.ERROR)

        # Disable transformers progress bars (Loading weights, etc.)
        try:
            transformers.logging.disable_progress_bar()
        except Exception:
            pass

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
            warnings.filterwarnings("ignore", message=".*LOAD REPORT.*")
            try:
                model = SentenceTransformer(self.model_name)
            finally:
                # Restore original log levels
                for name, level in saved_levels.items():
                    logging.getLogger(name).setLevel(level)
        return model
    
    def embed_tag(self, tag: str) -> np.ndarray:
        """
        Embed a single tag, using cache if available.
        
        Args:
            tag: Tag string to embed
            
        Returns:
            Numpy array of the embedding vector
        """
        tag_lower = tag.lower().strip()
        if tag_lower not in self._cache:
            # Replace underscores with spaces for better semantic encoding
            text = tag_lower.replace("_", " ")
            embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            self._cache[tag_lower] = embedding
        return self._cache[tag_lower]
    
    def embed_tags(self, tags: List[str]) -> Dict[str, np.ndarray]:
        """
        Embed multiple tags, using cache where available.
        
        Args:
            tags: List of tag strings to embed
            
        Returns:
            Dictionary mapping tag -> embedding vector
        """
        tags_lower = [t.lower().strip() for t in tags if t.strip()]
        
        # Find uncached tags
        uncached = [t for t in tags_lower if t not in self._cache]
        
        if uncached:
            # Batch encode uncached tags
            texts = [t.replace("_", " ") for t in uncached]
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            for tag, embedding in zip(uncached, embeddings):
                self._cache[tag] = embedding
        
        return {t: self._cache[t] for t in tags_lower}
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score between -1.0 and 1.0
        """
        # Vectors are already normalized by sentence-transformers,
        # so dot product equals cosine similarity
        return float(np.dot(a, b))
    
    def find_similar_tags(
        self,
        candidate_tags: List[str],
        existing_tags: List[str],
        threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find similar tags between candidate and existing tag sets using
        cosine similarity on embeddings.
        
        Args:
            candidate_tags: Tags from new memory candidates
            existing_tags: Tags from existing stored memories
            threshold: Similarity threshold override (uses self.similarity_threshold if None)
            
        Returns:
            List of (candidate_tag, existing_tag, similarity_score) tuples
            for pairs above the threshold, sorted by score descending
        """
        if not candidate_tags or not existing_tags:
            return []
        
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Embed all tags
        candidate_embeddings = self.embed_tags(candidate_tags)
        existing_embeddings = self.embed_tags(existing_tags)
        
        matches = []
        for c_tag, c_emb in candidate_embeddings.items():
            for e_tag, e_emb in existing_embeddings.items():
                if c_tag == e_tag:
                    # Exact match already handled elsewhere, skip
                    continue
                score = self.cosine_similarity(c_emb, e_emb)
                if score >= threshold:
                    matches.append((c_tag, e_tag, score))
        
        # Sort by similarity score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def get_similar_existing_tags(
        self,
        candidate_tags: List[str],
        existing_tags: List[str],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Get existing tags that are semantically similar to any candidate tag.
        
        This is a convenience method that returns just the existing tags
        (deduplicated) that match any candidate tag above the threshold.
        
        Args:
            candidate_tags: Tags from new memory candidates
            existing_tags: Tags from existing stored memories
            threshold: Similarity threshold override
            
        Returns:
            List of existing tags that are similar to candidate tags
        """
        matches = self.find_similar_tags(candidate_tags, existing_tags, threshold)
        # Deduplicate existing tags
        seen = set()
        result = []
        for _, e_tag, _ in matches:
            if e_tag not in seen:
                seen.add(e_tag)
                result.append(e_tag)
        return result
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
