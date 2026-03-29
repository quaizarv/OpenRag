import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .usage_log import _Timer, log_usage

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# OpenAI embeddings API limits
OPENAI_BATCH_LIMIT = 2048  # Max texts per batch
OPENAI_TOKEN_LIMIT = 100000  # Max tokens per batch (300K is limit, use 100K for safety with long docs)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts. Default implementation calls single method."""
        return [self.create_embedding(t) for t in texts]


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        # The OpenAI client uses an underlying HTTP client that may not be safe to share
        # across threads. Create one client per thread (important when building embeddings
        # with ThreadPoolExecutor).
        self._tls = threading.local()

    def _client(self) -> OpenAI:
        c = getattr(self._tls, "client", None)
        if c is None:
            c = OpenAI()
            self._tls.client = c
        return c

    def _normalize_text(self, text: str) -> str:
        """Normalize text for embedding: replace newlines, truncate, handle empty strings."""
        text = (text or "").replace("\n", " ").strip()
        # Truncate to ~8000 tokens (32000 chars) to stay under model's 8192 token limit
        max_chars = 32000
        if len(text) > max_chars:
            text = text[:max_chars]
        return text if text else " "

    # Embeddings can hit rate limits; use more patient exponential backoff.
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(12))
    def create_embedding(self, text):
        t = _Timer()
        text = self._normalize_text(text)
        resp = self._client().embeddings.create(input=[text], model=self.model)
        log_usage(
            kind="embeddings",
            model=self.model,
            usage=getattr(resp, "usage", None),
            duration_s=t.elapsed(),
        )
        return resp.data[0].embedding

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(12))
    def _embed_batch_chunk(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts (must be <= OPENAI_BATCH_LIMIT)."""
        t = _Timer()
        resp = self._client().embeddings.create(input=texts, model=self.model)
        log_usage(
            kind="embeddings",
            model=self.model,
            usage=getattr(resp, "usage", None),
            duration_s=t.elapsed(),
        )
        # OpenAI returns embeddings in same order as input
        return [d.embedding for d in resp.data]

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts using batch API calls.

        This is significantly faster than calling create_embedding() for each text
        individually since OpenAI's batch API reduces network round-trips.
        OpenAI charges per-token, not per-request, so there's no cost difference.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        # Normalize all texts
        normalized = [self._normalize_text(t) for t in texts]

        # Process in token-aware chunks to respect OpenAI's 300K token limit
        # Rough estimate: 1 token ≈ 4 chars for English text
        all_embeddings: List[List[float]] = []
        current_batch = []
        current_tokens = 0
        
        for text in normalized:
            # Estimate tokens (roughly 1 token per 4 chars)
            text_tokens = len(text) // 4 + 1
            
            # If adding this text would exceed token limit, process current batch first
            if current_tokens + text_tokens > OPENAI_TOKEN_LIMIT and current_batch:
                chunk_embeddings = self._embed_batch_chunk(current_batch)
                all_embeddings.extend(chunk_embeddings)
                current_batch = []
                current_tokens = 0
            
            # Also respect text count limit
            if len(current_batch) >= OPENAI_BATCH_LIMIT:
                chunk_embeddings = self._embed_batch_chunk(current_batch)
                all_embeddings.extend(chunk_embeddings)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        # Process remaining batch
        if current_batch:
            chunk_embeddings = self._embed_batch_chunk(current_batch)
            all_embeddings.extend(chunk_embeddings)

        return all_embeddings


class Qwen3EmbeddingModel(BaseEmbeddingModel):
    """
    Local embedding model using Qwen/Qwen3-Embedding-0.6B.

    Qwen3-Embedding is a decoder-only model that uses last-token pooling.
    Embedding dimension: 1024. Max sequence length: 32768.

    For retrieval queries, use get_query_embedding() to prepend the instruction
    prefix, which improves recall. For documents/passages, use create_embedding()
    or create_embeddings_batch() directly (no prefix needed).
    """

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIM = 1024
    MAX_LENGTH = 32768
    DEFAULT_QUERY_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    BATCH_SIZE = 32  # Texts per forward pass; reduce if OOM

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Qwen3EmbeddingModel requires 'torch' and 'transformers' packages."
            ) from e

        import torch

        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, torch_dtype="auto").to(device)
        self._model.eval()
        logging.info(f"Qwen3EmbeddingModel loaded on {device}: {model_name}")

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last non-padding token (Qwen3 uses right-padding by default)."""
        import torch

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F

        encoded = self._tokenizer(
            texts,
            max_length=self.MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encoded)

        embeddings = self._last_token_pool(
            outputs.last_hidden_state, encoded["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().tolist()

    def create_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            from tqdm import tqdm
            batches = range(0, len(texts), self.BATCH_SIZE)
            bar = tqdm(batches, desc="Embedding", unit="batch", total=len(batches))
        except ImportError:
            bar = range(0, len(texts), self.BATCH_SIZE)

        all_embeddings: List[List[float]] = []
        for i in bar:
            all_embeddings.extend(self._embed(texts[i : i + self.BATCH_SIZE]))
        return all_embeddings

    def get_query_embedding(self, query: str, instruction: str = None) -> List[float]:
        """
        Embed a retrieval query with the Qwen3 instruction prefix.

        Use this instead of create_embedding() when embedding user queries
        so the model knows to retrieve relevant passages.
        """
        instr = instruction or self.DEFAULT_QUERY_INSTRUCTION
        formatted = f"Instruct: {instr}\nQuery: {query}"
        return self.create_embedding(formatted)


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        # Import lazily to avoid hard dependency / version-coupling at module import time.
        # (Some environments will have transformers + huggingface_hub versions that break old sentence-transformers.)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SBertEmbeddingModel requires the 'sentence-transformers' package to be installed "
                "and compatible with your 'huggingface_hub' version."
            ) from e

        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed using sentence-transformers native batch support."""
        if not texts:
            return []
        # sentence-transformers encode() accepts a list and returns a numpy array
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]


def get_embedding_model(model_name: str = None) -> BaseEmbeddingModel:
    """
    Factory that returns an embedding model based on the EMBEDDING_MODEL env var.

    Supported values for EMBEDDING_MODEL:
      qwen3         — Qwen/Qwen3-Embedding-0.6B (default, local)
      openai        — OpenAI text-embedding-3-large
      openai-small  — OpenAI text-embedding-3-small
      sbert         — sentence-transformers/multi-qa-mpnet-base-cos-v1

    A custom HuggingFace model ID (e.g. "BAAI/bge-small-en-v1.5") is also
    accepted and will be loaded as a Qwen3EmbeddingModel (last-token pooling).
    For sentence-transformers models, set EMBEDDING_BACKEND=sbert alongside it.

    Args:
        model_name: Override the env var. If None, reads EMBEDDING_MODEL.
    """
    name = (model_name or os.environ.get("EMBEDDING_MODEL", "qwen3")).strip().lower()

    if name == "qwen3":
        return Qwen3EmbeddingModel()
    elif name == "openai":
        return OpenAIEmbeddingModel(model="text-embedding-3-large")
    elif name == "openai-small":
        return OpenAIEmbeddingModel(model="text-embedding-3-small")
    elif name == "sbert":
        return SBertEmbeddingModel()
    else:
        # Treat as a raw HuggingFace model ID
        backend = os.environ.get("EMBEDDING_BACKEND", "qwen3").strip().lower()
        if backend == "sbert":
            return SBertEmbeddingModel(model_name=model_name or name)
        return Qwen3EmbeddingModel(model_name=model_name or name)


# Dimension map for known models — used to set tree metadata correctly.
EMBEDDING_DIMENSIONS = {
    "qwen3": 1024,
    "openai": 3072,
    "openai-small": 1536,
    "sbert": 768,
}
