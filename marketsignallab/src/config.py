"""Central configuration for the deterministic V1 strategy and backtest."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    """Thresholds used to create market and position signals."""

    min_avg_volume_30d: float = 1_000_000.0
    max_relative_volume_30d: float = 2.5
    max_volatility_30d: float = 0.08
    spike_return_threshold: float = 0.20
    high_relative_volume: float = 2.0
    min_spike_profit: float = 0.10
    loss_recovery_trigger: float = -0.10
    loss_recovery_exit: float = 0.05
    hard_stop_loss: float = -0.30
    weekly_drop_threshold: float = -0.05
    weekly_relative_threshold: float = -0.03
    runner_stop_loss: float = -0.05
    reentry_min_return: float = 0.05
    reentry_max_return: float = 0.15

    def __post_init__(self) -> None:
        if self.min_avg_volume_30d < 0:
            raise ValueError("min_avg_volume_30d cannot be negative.")
        if not 0.0 < self.max_relative_volume_30d:
            raise ValueError("max_relative_volume_30d must be positive.")
        if not 0.0 <= self.max_volatility_30d:
            raise ValueError("max_volatility_30d cannot be negative.")
        if (
            not self.hard_stop_loss
            < self.loss_recovery_trigger
            < self.loss_recovery_exit
        ):
            raise ValueError(
                "Expected hard_stop_loss < loss_recovery_trigger < loss_recovery_exit."
            )
        if not 0.0 <= self.reentry_min_return < self.reentry_max_return:
            raise ValueError("Expected 0 <= reentry_min_return < reentry_max_return.")


@dataclass(frozen=True)
class BacktestConfig:
    """Portfolio, execution, cooldown, and watchlist settings."""

    starting_cash: float = 10_000.0
    max_positions: int = 5
    target_position_fraction: float = 0.20
    min_cash_buffer_fraction: float = 0.05
    compounder_sale_fraction: float = 0.90
    catastrophic_sale_fraction: float = 0.50
    loss_recovery_cooldown_days: int = 5
    weekly_drop_cooldown_days: int = 5
    runner_cooldown_days: int = 5
    max_watch_days: int = 20
    benchmark_ticker: str = "QQQ"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive.")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive.")
        if not 0.0 < self.target_position_fraction <= 1.0:
            raise ValueError("target_position_fraction must be in (0, 1].")
        if self.target_position_fraction * self.max_positions > 1.0 + 1e-12:
            raise ValueError(
                "target_position_fraction multiplied by max_positions cannot exceed 1."
            )
        if not 0.0 <= self.min_cash_buffer_fraction < 1.0:
            raise ValueError("min_cash_buffer_fraction must be in [0, 1).")
        for name, value in (
            ("compounder_sale_fraction", self.compounder_sale_fraction),
            ("catastrophic_sale_fraction", self.catastrophic_sale_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        for name, value in (
            ("loss_recovery_cooldown_days", self.loss_recovery_cooldown_days),
            ("weekly_drop_cooldown_days", self.weekly_drop_cooldown_days),
            ("runner_cooldown_days", self.runner_cooldown_days),
            ("max_watch_days", self.max_watch_days),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        if not self.benchmark_ticker.strip():
            raise ValueError("benchmark_ticker cannot be empty.")


DEFAULT_STRATEGY_CONFIG = StrategyConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
