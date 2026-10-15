"""Pair selection. Called ONLY with the training window."""
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from config import Config


@dataclass(frozen=True)
class Pair:
    y: str
    x: str
    beta: float       # hedge ratio: spread = log(y) - beta*log(x) - alpha
    alpha: float
    pvalue: float
    half_life: float


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1) fit: ds_t = a + b*s_{t-1}."""
    s = spread.dropna()
    lag, delta = s.shift(1).iloc[1:], s.diff().iloc[1:]
    b = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
    phi = 1.0 + b
    if not 0.0 < phi < 1.0:
        return np.inf
    return float(-np.log(2) / np.log(phi))


def select_pairs(train_prices: pd.DataFrame, cfg: Config) -> tuple[list[Pair], dict]:
    logp = np.log(train_prices)
    candidates, n_tests, n_coint = [], 0, 0
    for a, b in combinations(logp.columns, 2):
        n_tests += 1
        pvalue = coint(logp[a], logp[b], trend="c")[1]
        if pvalue >= cfg.pvalue_max:
            continue
        n_coint += 1
        alpha, beta = sm.OLS(logp[a], sm.add_constant(logp[b])).fit().params
        if beta <= 0:   # require a genuine long/short pair
            continue
        hl = half_life(logp[a] - beta * logp[b] - alpha)
        if not cfg.min_half_life <= hl <= cfg.max_half_life:
            continue
        candidates.append(Pair(a, b, float(beta), float(alpha), float(pvalue), hl))

    info = {"n_tests": n_tests, "n_coint": n_coint, "n_after_filters": len(candidates),
            "expected_false_positives": round(cfg.pvalue_max * n_tests, 1)}
    if not candidates:
        return [], info
    df = pd.DataFrame([asdict(p) for p in candidates])
    df["rank"] = df["pvalue"].rank() + df["half_life"].rank()   # low p and short half-life
    df = df.sort_values(["rank", "pvalue"]).head(cfg.top_n).drop(columns="rank")
    return [Pair(**row) for row in df.to_dict("records")], info
