"""Out-of-sample backtest with costs, metrics and charts."""
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from pairs import Pair
from strategy import compute_spread, generate_positions, rolling_zscore


@dataclass
class PairResult:
    pair: Pair
    daily: pd.DataFrame
    trades: list  # net pnl per round trip, as a fraction of the pair's gross capital


def net_returns(pos: pd.Series, spread_ret: pd.Series, beta: float, cost_bps: float):
    """pos[t] = position held after the close of day t, so it earns day t+1's return.

    Return is per $ of gross notional (1 long leg + |beta| short leg).
    Cost = cost_bps on every unit of notional traded, charged on the day the position changes.
    """
    gross = pos.shift(1).fillna(0.0) * spread_ret / (1.0 + abs(beta))
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    net = gross - turnover * cost_bps / 1e4
    return gross, turnover, net


def extract_trades(pos: pd.Series, net: pd.Series) -> list:
    """Net pnl per round trip (entry cost + holding pnl + exit cost). No flips by construction."""
    trades, cur, prev = [], None, 0.0
    for p, n in zip(pos.to_numpy(), net.to_numpy()):
        if p != prev:
            if cur is not None:
                trades.append(cur + n)
                cur = None
            if p != 0:
                cur = n
        elif cur is not None:
            cur += n
        prev = p
    return trades


def backtest_pair(prices: pd.DataFrame, pair: Pair, cfg: Config,
                  entry_ok: pd.Series | None = None) -> PairResult:
    spread = compute_spread(prices, pair)                  # training-window beta
    z = rolling_zscore(spread, cfg.z_window)               # trailing; warm-up uses train tail
    test = prices.index[prices.index >= pd.Timestamp(cfg.test_start)]
    sig = generate_positions(z.loc[test], cfg, entry_ok)   # starts flat on first test day
    pos = sig["target"].shift(cfg.exec_lag).fillna(0.0).astype(float)
    pos.iloc[-1] = 0.0                                     # liquidate at end of test window
    gross, turnover, net = net_returns(pos, spread.diff().loc[test], pair.beta, cfg.cost_bps)
    daily = pd.DataFrame({"z": z.loc[test], "spread": spread.loc[test], "target": sig["target"],
                          "pos": pos, "gross": gross, "turnover": turnover, "net": net})
    return PairResult(pair, daily, extract_trades(pos, net))


def perf_stats(net: pd.Series, turnover: pd.Series, trades: list) -> dict:
    equity = (1 + net).cumprod()
    sd = net.std()
    return {
        "total_return": equity.iloc[-1] - 1,
        "sharpe": net.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "trades": len(trades),
        "win_rate": float(np.mean(np.array(trades) > 0)) if trades else np.nan,
        "turnover_per_year": turnover.sum() / (len(net) / 252),
    }


def portfolio_returns(results: list[PairResult]) -> pd.Series:
    """Equal-weight: each pair gets 1/N of capital."""
    return pd.concat([r.daily["net"] for r in results], axis=1).mean(axis=1)


def summarize(results: list[PairResult]) -> pd.DataFrame:
    rows = {f"{r.pair.y}/{r.pair.x}": perf_stats(r.daily["net"], r.daily["turnover"], r.trades)
            for r in results}
    turnover = pd.concat([r.daily["turnover"] for r in results], axis=1).mean(axis=1)
    all_trades = [t for r in results for t in r.trades]
    rows["PORTFOLIO"] = perf_stats(portfolio_returns(results), turnover, all_trades)
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_equity(base: list[PairResult], ml: list[PairResult] | None, path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in base:
        ax.plot((1 + r.daily["net"]).cumprod(), lw=0.9, alpha=0.6, label=f"{r.pair.y}/{r.pair.x}")
    ax.plot((1 + portfolio_returns(base)).cumprod(), color="black", lw=2.5, label="Portfolio")
    if ml:
        ax.plot((1 + portfolio_returns(ml)).cumprod(), color="red", lw=2, ls="--",
                label="Portfolio + ML filter")
    ax.axhline(1.0, color="grey", lw=0.5)
    ax.set_title("Out-of-sample equity curves (net of costs)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pair(res: PairResult, cfg: Config, path: str) -> None:
    d = res.daily
    prev = d["pos"].shift(1).fillna(0.0)
    marks = {"long entry": ((d["pos"] == 1) & (prev == 0), "^", "green"),
             "short entry": ((d["pos"] == -1) & (prev == 0), "v", "red"),
             "exit": ((d["pos"] == 0) & (prev != 0), "x", "black")}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(d["spread"], lw=1)
    ax1.set_title(f"{res.pair.y}/{res.pair.x}: spread (beta={res.pair.beta:.2f}), test window")
    ax2.plot(d["z"], lw=1)
    for level, style in [(cfg.entry_z, "--"), (cfg.exit_z, ":"), (cfg.stop_z, "-.")]:
        for s in (1, -1):
            ax2.axhline(s * level, color="grey", ls=style, lw=0.8)
    for label, (mask, marker, color) in marks.items():
        ax1.scatter(d.index[mask], d["spread"][mask], marker=marker, c=color, s=50, label=label, zorder=3)
        ax2.scatter(d.index[mask], d["z"][mask], marker=marker, c=color, s=50, zorder=3)
    ax2.set_title("Rolling z-score")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
