"""Round-boundary recovery for sequential CFD federated simulations."""
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


class RoundCheckpoint:
    def __init__(self, args, method, alpha):
        self.args = args
        self.path = Path(args.output_dir) / f"latest_{method}_alpha_{alpha:g}.pt"
        self.history_path = self.path.with_suffix(".csv")
        self.progress_path = self.path.with_suffix(".json")
        self.started = time.perf_counter()
        self.elapsed = 0.0
        self.counts = dict(optimizer_steps=0, client_updates=0,
                           model_uploads=0, correction_uploads=0, upload_bytes=0)
        self.budget_history = []

    def load(self, rng=None):
        if not self.args.resume:
            return None
        # Only load checkpoints produced by this project from trusted storage.
        saved = torch.load(self.args.resume, map_location="cpu", weights_only=False)
        if saved["signature"] != self.args._run_signature or saved["filename"] != self.path.name:
            raise ValueError("Checkpoint configuration/data/source differs from this run")
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if self.args.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if rng is not None:
            rng.bit_generator.state = saved["selection_rng"]
        self.counts = saved["counts"]
        self.budget_history = saved["budget_history"]
        self.elapsed = saved["elapsed_seconds"]
        self.started = time.perf_counter()
        print(f"Resumed {self.path.name} at round {saved['round']}", flush=True)
        return saved

    def client(self, batches):
        self.counts["optimizer_steps"] += batches
        self.counts["client_updates"] += 1

    def uploads(self, state, count, correction=False):
        self.counts["correction_uploads" if correction else "model_uploads"] += count
        size = sum(v.numel() * v.element_size() for v in state.values() if v.is_floating_point())
        self.counts["upload_bytes"] += count * size

    def save(self, round_idx, state, history, rng=None, final_errors=None):
        elapsed = self.elapsed + time.perf_counter() - self.started
        self.budget_history.append(dict(round=round_idx, elapsed_seconds=elapsed, **self.counts))
        saved = dict(
            version=1, filename=self.path.name, signature=self.args._run_signature,
            round=round_idx, state=state, history=history, final_errors=final_errors,
            selection_rng=None if rng is None else rng.bit_generator.state,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.args.device == "cuda" else None,
            counts=self.counts, budget_history=self.budget_history, elapsed_seconds=elapsed,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".pt.tmp")
        torch.save(saved, tmp)
        os.replace(tmp, self.path)
        # The checkpoint is authoritative if interrupted while exporting metrics.
        self._csv(self.history_path, history)
        self._csv(self.path.with_name(self.path.stem + "_budget.csv"), self.budget_history)
        tmp = self.progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(round=round_idx, elapsed_seconds=elapsed,
                                      checkpoint=str(self.path), **self.counts), indent=2))
        os.replace(tmp, self.progress_path)
        print(f"Saved round {round_idx}: {self.counts}", flush=True)

    @staticmethod
    def _csv(path, rows):
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
