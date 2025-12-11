"""
logger.py

Unified Logging System for Infinity Dual Hybrid v2.0.

Supports multiple backends:
- CSV logging
- JSONL logging
- TensorBoard logging (optional)
- Console logging
"""

import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False


@dataclass
class LoggerConfig:
    """Configuration for the unified logger."""
    log_dir: str = "logs"
    experiment_name: Optional[str] = None
    use_csv: bool = True
    use_jsonl: bool = True
    use_tensorboard: bool = False
    use_console: bool = True
    console_interval: int = 10
    flush_interval: int = 1


class UnifiedLogger:
    """
    Unified logging system with multiple backends.

    Logs training metrics to CSV, JSONL, TensorBoard, and console.
    """

    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg
        self.step = 0
        self.start_time = time.time()

        # Create experiment directory
        if cfg.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg.experiment_name = f"run_{timestamp}"

        self.log_path = Path(cfg.log_dir) / cfg.experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Initialize backends
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None
        self._tb_writer = None
        self._csv_fields: Optional[List[str]] = None

        if cfg.use_csv:
            self._init_csv()

        if cfg.use_jsonl:
            self._init_jsonl()

        if cfg.use_tensorboard and HAS_TENSORBOARD:
            self._init_tensorboard()

    def _init_csv(self) -> None:
        """Initialize CSV logger."""
        csv_path = self.log_path / "metrics.csv"
        self._csv_file = open(csv_path, "w", newline="")

    def _init_jsonl(self) -> None:
        """Initialize JSONL logger."""
        jsonl_path = self.log_path / "metrics.jsonl"
        self._jsonl_file = open(jsonl_path, "w")

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard logger."""
        if HAS_TENSORBOARD:
            tb_path = self.log_path / "tensorboard"
            self._tb_writer = SummaryWriter(log_dir=str(tb_path))

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (uses internal counter if not provided)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

        # Add metadata
        elapsed = time.time() - self.start_time
        full_metrics = {
            "step": self.step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            **metrics,
        }

        # Log to each backend
        if self.cfg.use_csv:
            self._log_csv(full_metrics)

        if self.cfg.use_jsonl:
            self._log_jsonl(full_metrics)

        if self.cfg.use_tensorboard and self._tb_writer is not None:
            self._log_tensorboard(metrics)

        if self.cfg.use_console and self.step % self.cfg.console_interval == 0:
            self._log_console(full_metrics)

        # Periodic flush
        if self.step % self.cfg.flush_interval == 0:
            self.flush()

    def _log_csv(self, metrics: Dict[str, Any]) -> None:
        """Log to CSV file."""
        if self._csv_file is None:
            return

        # Initialize CSV writer with fields on first log
        if self._csv_writer is None:
            self._csv_fields = list(metrics.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fields
            )
            self._csv_writer.writeheader()

        # Filter metrics to known fields
        row = {k: metrics.get(k, "") for k in self._csv_fields}
        self._csv_writer.writerow(row)

    def _log_jsonl(self, metrics: Dict[str, Any]) -> None:
        """Log to JSONL file."""
        if self._jsonl_file is None:
            return

        # Convert non-serializable types
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                if v != v:  # NaN check
                    clean_metrics[k] = None
                else:
                    clean_metrics[k] = v
            else:
                clean_metrics[k] = v

        self._jsonl_file.write(json.dumps(clean_metrics) + "\n")

    def _log_tensorboard(self, metrics: Dict[str, Any]) -> None:
        """Log to TensorBoard."""
        if self._tb_writer is None:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._tb_writer.add_scalar(key, value, self.step)

    def _log_console(self, metrics: Dict[str, Any]) -> None:
        """Log to console."""
        step = metrics.get("step", self.step)
        elapsed = metrics.get("elapsed_seconds", 0)

        # Format key metrics
        parts = [f"[{step:6d}]"]

        # Add common metrics if present
        if "mean_reward" in metrics:
            parts.append(f"reward={metrics['mean_reward']:.2f}")
        if "mean_return" in metrics:
            parts.append(f"return={metrics['mean_return']:.2f}")
        if "policy_loss" in metrics:
            parts.append(f"ploss={metrics['policy_loss']:.4f}")
        if "value_loss" in metrics:
            parts.append(f"vloss={metrics['value_loss']:.4f}")
        if "entropy" in metrics:
            parts.append(f"ent={metrics['entropy']:.4f}")
        if "kl" in metrics:
            parts.append(f"kl={metrics['kl']:.4f}")
        if "grad_norm" in metrics:
            parts.append(f"gnorm={metrics['grad_norm']:.2f}")

        # Add timing
        parts.append(f"({elapsed:.1f}s)")

        print(" ".join(parts))

    def flush(self) -> None:
        """Flush all file handles."""
        if self._csv_file is not None:
            self._csv_file.flush()
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def close(self) -> None:
        """Close all file handles."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def __enter__(self) -> "UnifiedLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_log_path(self) -> Path:
        """Get the log directory path."""
        return self.log_path


def create_logger(
    log_dir: str = "logs",
    experiment_name: Optional[str] = None,
    use_tensorboard: bool = False,
) -> UnifiedLogger:
    """
    Factory function for creating a logger.

    Args:
        log_dir: Base directory for logs
        experiment_name: Name for this experiment
        use_tensorboard: Enable TensorBoard logging

    Returns:
        Configured UnifiedLogger instance
    """
    cfg = LoggerConfig(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
    )
    return UnifiedLogger(cfg)
