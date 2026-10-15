"""Spread, rolling z-score, and the entry/exit state machine."""
import numpy as np
import pandas as pd

from config import Config
from pairs import Pair


def compute_spread(prices: pd.DataFrame, pair: Pair) -> pd.Series:
    """Uses the hedge ratio fitted on the training window. Nothing is refit here."""
    return np.log(prices[pair.y]) - pair.beta * np.log(prices[pair.x]) - pair.alpha


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Trailing window only: z[t] depends on spread[:t+1] and nothing later."""
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def generate_positions(z: pd.Series, cfg: Config, entry_ok: pd.Series | None = None) -> pd.DataFrame:
    """Target spread position decided at each close: +1 long spread, -1 short, 0 flat.

    Long spread at z < -entry_z, short at z > entry_z, exit at |z| < exit_z,
    stop at |z| > stop_z or after max_hold_days. After a stop/timeout, no re-entry
    until the spread has come back inside the exit band.
    """
    ok = (np.ones(len(z), dtype=bool) if entry_ok is None
          else entry_ok.reindex(z.index).fillna(False).to_numpy(dtype=bool))
    pos, days_held, blocked = 0, 0, False
    targets, events = [], []
    for t, zt in enumerate(z.to_numpy()):
        event = ""
        if np.isnan(zt):
            pass
        elif pos != 0:
            days_held += 1
            if abs(zt) < cfg.exit_z:
                pos, event = 0, "exit"
            elif abs(zt) > cfg.stop_z:
                pos, blocked, event = 0, True, "stop"
            elif days_held >= cfg.max_hold_days:
                pos, blocked, event = 0, True, "timeout"
        elif blocked:
            if abs(zt) < cfg.exit_z:
                blocked = False
        elif cfg.entry_z <= abs(zt) < cfg.stop_z and ok[t]:
            pos = -int(np.sign(zt))          # z > 0 -> short spread
            days_held = 0
            event = "enter_long" if pos == 1 else "enter_short"
        targets.append(pos)
        events.append(event)
    out = pd.DataFrame({"target": targets, "event": events}, index=z.index)
    out["signal"] = out["target"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
    return out


def describe_signal(signal: str, pair: Pair) -> str:
    y, x, b = pair.y, pair.x, abs(pair.beta)
    if signal == "BUY":
        return f"Long spread: per $1 of {y} BOUGHT, SHORT ${b:.2f} of {x}"
    if signal == "SELL":
        return f"Short spread: per $1 of {y} SHORTED, BUY ${b:.2f} of {x}"
    return "No position: stay flat (close any open position)"


def latest_signals(prices: pd.DataFrame, pairs: list[Pair], cfg: Config,
                   entry_filters: dict | None = None) -> pd.DataFrame:
    """Signal as of the latest close. Execute at the next session."""
    rows = []
    for p in pairs:
        z = rolling_zscore(compute_spread(prices, p), cfg.z_window)
        ok = None if entry_filters is None else entry_filters.get((p.y, p.x))
        sig = generate_positions(z.loc[cfg.test_start:], cfg, ok)
        last = sig.iloc[-1]
        rows.append({"pair": f"{p.y}/{p.x}", "as_of": sig.index[-1].date(),
                     "z": round(float(z.iloc[-1]), 2), "signal": last["signal"],
                     "event": last["event"], "meaning": describe_signal(last["signal"], p)})
    return pd.DataFrame(rows)
