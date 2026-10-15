"""All tunable settings live here. Logic modules read from Config, never hardcode."""
from dataclasses import dataclass, field

BANKS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW", "COF",
         "BK", "STT", "FITB", "RF", "KEY", "HBAN", "CFG", "MTB", "NTRS", "ZION", "CMA"]
ENERGY = ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "MPC", "VLO", "WMB",
          "KMI", "OKE", "HAL", "DVN", "FANG", "BKR", "CTRA", "APA", "EQT"]


@dataclass
class Config:
    # data
    universe: list = field(default_factory=lambda: list(BANKS))
    start: str = "2019-01-01"
    train_end: str = "2022-12-31"   # pair selection + hedge ratios use ONLY data <= this date
    test_start: str = "2023-01-01"  # all reported results use ONLY data >= this date
    end: str | None = None          # None = today
    data_dir: str = "data"
    output_dir: str = "outputs"
    # pair selection
    pvalue_max: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 60.0
    top_n: int = 5
    # strategy
    z_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold_days: int = 40
    exec_lag: int = 1               # signal at close t -> filled at close t+exec_lag
    # costs
    cost_bps: float = 5.0           # per side, per leg, on traded notional
    # ML filter
    ml_horizon: int = 20
    ml_threshold: float | None = None  # None = training-set base rate
    # live (optional)
    live_capital_usd: float = 10_000.0
