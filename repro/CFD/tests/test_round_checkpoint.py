"""Verify interrupted runs reproduce uninterrupted stochastic updates."""
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import ppbc_drivaernetpp_transformer_easy as experiment
from util.round_checkpoint import RoundCheckpoint


class RecoveryTests(unittest.TestCase):
    def test_round_boundary_recovery(self):
        for method in ("fedavg", "ppbc"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(output_dir=directory + "/full", resume="", device="cpu",
                    _run_signature="fixture", seed=42, rounds=3, eval_every=1,
                    aggregation="equal", ppbc_iterations=2, epoch_k=2, iter_k=1,
                    theta=0.25, gamma=1.0, q_m=1.0)
                initial = {"w": torch.ones(2)}

                def train(state, *unused):
                    update = torch.rand(2) + float(np.random.random()) + random.random()
                    return {"w": update * 0.01 - state["w"] * 0.02}, 1.0, 2

                def evaluate(state, dataset, records, args, cfg, fabric, dm, method, alpha, round_idx, *rest):
                    # Evaluation consumes RNG too, as the real model constructor does.
                    torch.rand(1)
                    return [dict(method=method, round=round_idx, body="all", value=float(state['w'].sum()))]

                run = getattr(experiment, "run_" + method)
                def execute():
                    experiment.set_seed(42)
                    return run(initial, None, None, [], [[0], [1]], args, None, None, None, 0.1)

                with patch.object(experiment, "train_client", train), patch.object(experiment, "evaluate_state", evaluate), patch.object(experiment, "print_round"):
                    expected_history = execute()
                    name = f"latest_{method}_alpha_0.1.pt"
                    expected = torch.load(Path(args.output_dir) / name, weights_only=False)
                    args.output_dir = directory + "/resumed"
                    original_save = RoundCheckpoint.save

                    def interrupt(checkpoint, round_idx, *a, **kw):
                        original_save(checkpoint, round_idx, *a, **kw)
                        if round_idx == 1:
                            raise RuntimeError("simulated interruption")

                    with patch.object(RoundCheckpoint, "save", interrupt):
                        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                            execute()
                    args.resume = str(Path(args.output_dir) / name)
                    actual_history = execute()
                    actual = torch.load(args.resume, weights_only=False)
                    self.assertEqual(expected_history, actual_history)
                    self.assertTrue(torch.equal(expected["state"]["w"], actual["state"]["w"]))
                    self.assertEqual(expected["counts"], actual["counts"])
                    self.assertEqual(len(actual["budget_history"]), 4)
                    if method == "ppbc":
                        for client in expected["final_errors"]:
                            self.assertTrue(torch.equal(expected["final_errors"][client]["w"],
                                                        actual["final_errors"][client]["w"]))
                        self.assertEqual(actual["counts"]["optimizer_steps"], 24)
                        self.assertEqual(actual["counts"]["model_uploads"], 6)
                        self.assertEqual(actual["counts"]["correction_uploads"], 6)
                    else:
                        self.assertEqual(actual["counts"]["optimizer_steps"], 12)
                        self.assertEqual(actual["counts"]["model_uploads"], 6)
                    args._run_signature = "changed configuration"
                    with self.assertRaises(ValueError):
                        execute()


if __name__ == "__main__":
    unittest.main()
