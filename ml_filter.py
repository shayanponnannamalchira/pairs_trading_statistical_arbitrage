"""Optional: logistic-regression filter on entries, trained on the training window only."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from pairs import Pair
from strategy import compute_spread, rolling_zscore

FEATURES = ["z", "abs_z", "dz_5", "vol_20", "half_life"]


def pair_features(prices: pd.DataFrame, pair: Pair, cfg: Config):
    spread = compute_spread(prices, pair)
    z = rolling_zscore(spread, cfg.z_window)
    feats = pd.DataFrame({"z": z, "abs_z": z.abs(), "dz_5": z.diff(5),
                          "vol_20": spread.diff().rolling(20).std(),
                          "half_life": pair.half_life})
    return feats[FEATURES], z


def label_events(z: pd.Series, cfg: Config) -> pd.Series:
    """1 if |z| falls below exit_z within ml_horizon days BEFORE breaching stop_z, else 0.
    Days whose horizon runs past the end of z are dropped, so no label peeks beyond the window."""
    zv, out = z.to_numpy(), {}
    for t in range(len(zv) - cfg.ml_horizon):
        if np.isnan(zv[t]) or not (cfg.entry_z <= abs(zv[t]) < cfg.stop_z):
            continue
        fut = np.abs(zv[t + 1: t + 1 + cfg.ml_horizon])
        rev, stp = np.where(fut < cfg.exit_z)[0], np.where(fut > cfg.stop_z)[0]
        r = rev[0] if len(rev) else np.inf
        s = stp[0] if len(stp) else np.inf
        out[z.index[t]] = int(r < s)
    return pd.Series(out, dtype=float)


@dataclass
class EntryFilter:
    model: object
    threshold: float
    n_train: int
    base_rate: float

    def entry_ok(self, prices: pd.DataFrame, pair: Pair, cfg: Config) -> pd.Series:
        feats, _ = pair_features(prices, pair, cfg)
        ok = pd.Series(False, index=feats.index)
        valid = feats.dropna().index
        ok.loc[valid] = self.model.predict_proba(feats.loc[valid])[:, 1] >= self.threshold
        return ok


def fit_entry_filter(train_prices: pd.DataFrame, pairs: list[Pair], cfg: Config) -> EntryFilter | None:
    xs, ys = [], []
    for p in pairs:
        feats, z = pair_features(train_prices, p, cfg)
        labels = label_events(z, cfg)
        feats = feats.loc[labels.index].dropna()
        xs.append(feats)
        ys.append(labels.loc[feats.index])
    X, y = pd.concat(xs), pd.concat(ys)
    if len(y) < 50 or y.nunique() < 2:
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    thr = cfg.ml_threshold if cfg.ml_threshold is not None else float(y.mean())
    return EntryFilter(model, thr, len(y), float(y.mean()))
