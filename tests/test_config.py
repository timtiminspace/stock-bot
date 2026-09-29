"""Validation tests for strategy and portfolio configuration."""

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "marketsignallab" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import BacktestConfig, StrategyConfig


class ConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        self.assertEqual(BacktestConfig().max_positions, 5)
        self.assertEqual(StrategyConfig().loss_recovery_exit, 0.05)

    def test_position_targets_cannot_require_leverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed 1"):
            BacktestConfig(max_positions=6, target_position_fraction=0.20)

    def test_loss_recovery_thresholds_must_be_ordered(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard_stop_loss"):
            StrategyConfig(hard_stop_loss=-0.05)


if __name__ == "__main__":
    unittest.main()
