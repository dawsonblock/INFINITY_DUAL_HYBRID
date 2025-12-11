"""
ltm.py

Long-Term Memory (LTM) System with FAISS support.

Provides episodic memory storage and retrieval:
- SimpleLTM: In-memory cosine similarity (fallback when FAISS unavailable)
- FaissIVFPQLTM: Scalable FAISS IVF-PQ index for large memory stores
- AsyncLTMWriter: Background thread for non-blocking memory writes

Usage:
    ltm = build_ltm(cfg)
    ltm.store(keys, values)  # Add memories
    retrieved = ltm.retrieve(queries, top_k=8)  # Query memories
"""

import threading
from queue import Queue
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LTMConfig

# Optional FAISS import
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False


class SimpleLTM(nn.Module):
    """
    Simple in-memory LTM with cosine similarity retrieval.

    Used as fallback when FAISS is not available.
    Suitable for smaller memory sizes (< 10k entries).
    """

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("keys", torch.empty(0, cfg.d_key))
        self.register_buffer("values", torch.empty(0, cfg.d_value))

    @property
    def size(self) -> int:
        return self.keys.shape[0]

    @torch.no_grad()
    def store(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Store key-value pairs in memory.

        Args:
            keys: [N, d_key] embeddings to index
            values: [N, d_value] associated values
        """
        if keys.numel() == 0:
            return
        keys = keys.detach().to(self.keys.device)
        values = values.detach().to(self.values.device)

        self.keys = torch.cat([self.keys, keys], dim=0)
        self.values = torch.cat([self.values, values], dim=0)

        # Enforce max size (FIFO eviction)
        if self.keys.shape[0] > self.cfg.max_size:
            self.keys = self.keys[-self.cfg.max_size:]
            self.values = self.values[-self.cfg.max_size:]

    @torch.no_grad()
    def retrieve(
        self,
        queries: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Retrieve values by cosine similarity.

        Args:
            queries: [B, d_key] query embeddings
            top_k: Number of neighbors (default: cfg.top_k)
        Returns:
            [B, d_value] weighted sum of retrieved values
        """
        top_k = top_k or self.cfg.top_k
        B = queries.shape[0]

        if self.keys.shape[0] == 0:
            return torch.zeros(B, self.cfg.d_value, device=queries.device)

        # Normalize for cosine similarity
        k_norm = F.normalize(self.keys, dim=-1)
        q_norm = F.normalize(queries, dim=-1)

        # Compute similarities
        sim = q_norm @ k_norm.t()  # [B, N]

        # Top-k retrieval
        k_eff = min(top_k, sim.shape[1])
        scores, indices = torch.topk(sim, k_eff, dim=-1)

        # Softmax weighting
        weights = F.softmax(scores, dim=-1)  # [B, k_eff]

        # Weighted sum of values
        retrieved_values = self.values[indices]  # [B, k_eff, d_value]
        output = (retrieved_values * weights.unsqueeze(-1)).sum(dim=1)  # [B, d_value]

        return output

    def clear(self) -> None:
        """Clear all stored memories."""
        self.keys = torch.empty(0, self.cfg.d_key, device=self.keys.device)
        self.values = torch.empty(0, self.cfg.d_value, device=self.values.device)

    def state_dict_ltm(self) -> dict:
        """Get LTM state for saving."""
        return {
            "keys": self.keys.cpu(),
            "values": self.values.cpu(),
        }

    def load_state_dict_ltm(self, state: dict) -> None:
        """Load LTM state."""
        self.keys = state["keys"].to(self.keys.device)
        self.values = state["values"].to(self.values.device)


class FaissIVFPQLTM(nn.Module):
    """
    FAISS-backed LTM with IVF-PQ index for scalable retrieval.

    Uses Inverted File Index with Product Quantization for
    efficient approximate nearest neighbor search on large
    memory stores (100k+ entries).
    """

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        if not HAS_FAISS:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")

        self.cfg = cfg
        self.register_buffer("keys_buf", torch.empty(0, cfg.d_key))
        self.register_buffer("values_buf", torch.empty(0, cfg.d_value))

        # Create FAISS index
        self.quantizer = faiss.IndexFlatL2(cfg.d_key)
        self.index = faiss.IndexIVFPQ(
            self.quantizer,
            cfg.d_key,
            cfg.nlist,
            cfg.m,
            8  # bits per subquantizer
        )
        self.index.nprobe = cfg.nprobe
        self._trained = False

    @property
    def size(self) -> int:
        return self.keys_buf.shape[0]

    @torch.no_grad()
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from buffer."""
        self.index.reset()

        if self.keys_buf.shape[0] == 0:
            self._trained = False
            return

        keys_np = self.keys_buf.cpu().numpy().astype("float32")

        # Train index if needed
        if not self.index.is_trained:
            # Need at least nlist vectors to train
            if keys_np.shape[0] >= self.cfg.nlist:
                self.index.train(keys_np)
            else:
                # Fall back to flat index behavior
                pass

        if self.index.is_trained:
            self.index.add(keys_np)
            self._trained = True

    @torch.no_grad()
    def store(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Store key-value pairs."""
        if keys.numel() == 0:
            return

        keys = keys.detach()
        values = values.detach()

        self.keys_buf = torch.cat([self.keys_buf, keys.to(self.keys_buf.device)], dim=0)
        self.values_buf = torch.cat([self.values_buf, values.to(self.values_buf.device)], dim=0)

        # Enforce max size
        if self.keys_buf.shape[0] > self.cfg.max_size:
            self.keys_buf = self.keys_buf[-self.cfg.max_size:]
            self.values_buf = self.values_buf[-self.cfg.max_size:]

        self._rebuild_index()

    @torch.no_grad()
    def retrieve(
        self,
        queries: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Retrieve values using FAISS index."""
        top_k = top_k or self.cfg.top_k
        B = queries.shape[0]

        if self.keys_buf.shape[0] == 0 or not self._trained:
            return torch.zeros(B, self.cfg.d_value, device=queries.device)

        # Query FAISS
        q_np = queries.cpu().numpy().astype("float32")
        k_eff = min(top_k, self.index.ntotal)

        if k_eff == 0:
            return torch.zeros(B, self.cfg.d_value, device=queries.device)

        distances, indices = self.index.search(q_np, k_eff)

        # Convert to tensors
        indices_t = torch.from_numpy(indices).to(self.values_buf.device)
        distances_t = torch.from_numpy(distances).to(self.values_buf.device)

        # Handle invalid indices (-1 from FAISS)
        valid_mask = indices_t >= 0
        indices_t = indices_t.clamp(min=0)

        # Softmax weighting (negative distance = higher similarity)
        weights = F.softmax(-distances_t, dim=-1)
        weights = weights * valid_mask.float()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum
        retrieved_values = self.values_buf[indices_t]
        output = (retrieved_values * weights.unsqueeze(-1)).sum(dim=1)

        return output

    def clear(self) -> None:
        """Clear all memories."""
        self.keys_buf = torch.empty(0, self.cfg.d_key, device=self.keys_buf.device)
        self.values_buf = torch.empty(0, self.cfg.d_value, device=self.values_buf.device)
        self.index.reset()
        self._trained = False

    def state_dict_ltm(self) -> dict:
        """Get LTM state."""
        return {
            "keys": self.keys_buf.cpu(),
            "values": self.values_buf.cpu(),
        }

    def load_state_dict_ltm(self, state: dict) -> None:
        """Load LTM state and rebuild index."""
        self.keys_buf = state["keys"].to(self.keys_buf.device)
        self.values_buf = state["values"].to(self.values_buf.device)
        self._rebuild_index()


class AsyncLTMWriter:
    """
    Asynchronous writer for non-blocking LTM updates.

    Batches writes and processes them in a background thread
    to avoid blocking the training loop.
    """

    def __init__(self, ltm: nn.Module, batch_size: int = 64):
        self.ltm = ltm
        self.batch_size = batch_size
        self._queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background writer thread."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._queue.put(None)  # Sentinel to unblock
        self._thread.join(timeout=5.0)
        self._thread = None

    def _writer_loop(self) -> None:
        """Background loop that processes queued writes."""
        keys_batch = []
        values_batch = []

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except Exception:
                # Timeout - flush if we have pending items
                if keys_batch:
                    self._flush(keys_batch, values_batch)
                    keys_batch, values_batch = [], []
                continue

            if item is None:
                break

            keys, values = item
            keys_batch.append(keys)
            values_batch.append(values)

            if len(keys_batch) >= self.batch_size:
                self._flush(keys_batch, values_batch)
                keys_batch, values_batch = [], []

        # Final flush
        if keys_batch:
            self._flush(keys_batch, values_batch)

    def _flush(self, keys_batch: list, values_batch: list) -> None:
        """Flush batched writes to LTM."""
        if not keys_batch:
            return
        keys = torch.cat(keys_batch, dim=0)
        values = torch.cat(values_batch, dim=0)
        self.ltm.store(keys, values)

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Queue a write operation."""
        self._queue.put((keys.detach().cpu(), values.detach().cpu()))

    def flush_sync(self) -> None:
        """Synchronously flush all pending writes."""
        # Drain the queue
        items = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    items.append(item)
            except Exception:
                break

        if items:
            keys = torch.cat([k for k, v in items], dim=0)
            values = torch.cat([v for k, v in items], dim=0)
            self.ltm.store(keys, values)


class LTMWrapper(nn.Module):
    """
    Wrapper that provides unified interface and optional async writing.
    """

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        self.cfg = cfg

        # Create underlying LTM
        if cfg.use_faiss and HAS_FAISS:
            self._ltm = FaissIVFPQLTM(cfg)
        else:
            self._ltm = SimpleLTM(cfg)

        # Optional async writer
        self._async_writer: Optional[AsyncLTMWriter] = None
        if cfg.use_async_writer:
            self._async_writer = AsyncLTMWriter(self._ltm, cfg.write_batch_size)
            self._async_writer.start()

    @property
    def size(self) -> int:
        return self._ltm.size

    def store(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Store memories (async if enabled)."""
        if self._async_writer is not None:
            self._async_writer.write(keys, values)
        else:
            self._ltm.store(keys, values)

    def retrieve(self, queries: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """Retrieve memories."""
        return self._ltm.retrieve(queries, top_k)

    def clear(self) -> None:
        """Clear all memories."""
        if self._async_writer is not None:
            self._async_writer.flush_sync()
        self._ltm.clear()

    def state_dict_ltm(self) -> dict:
        """Get LTM state."""
        if self._async_writer is not None:
            self._async_writer.flush_sync()
        return self._ltm.state_dict_ltm()

    def load_state_dict_ltm(self, state: dict) -> None:
        """Load LTM state."""
        self._ltm.load_state_dict_ltm(state)

    def shutdown(self) -> None:
        """Shutdown async writer if active."""
        if self._async_writer is not None:
            self._async_writer.stop()


def build_ltm(cfg: LTMConfig) -> LTMWrapper:
    """Build LTM from config."""
    return LTMWrapper(cfg)
