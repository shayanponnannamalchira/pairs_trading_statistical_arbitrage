"""Offline sanity tests: python tests.py"""
import numpy as np
import pandas as pd

from backtest import extract_trades, net_returns
from config import Config
from pairs import half_life
from strategy import generate_positions, rolling_zscore


def test_zscore_is_causal():
    s = pd.Series(np.random.default_rng(0).normal(size=300).cumsum())
    pd.testing.assert_series_equal(rolling_zscore(s, 60).iloc[:200], rolling_zscore(s.iloc[:200], 60))


def test_state_machine():
    z = pd.Series([0, -2.5, -1.0, 0.2, 0, 2.5, 4.5, 3.0, 2.5, 0.1, 2.5])
    out = generate_positions(z, Config())
    assert list(out["target"]) == [0, 1, 1, 0, 0, -1, 0, 0, 0, 0, -1]  # stop-out blocks re-entry


def test_costs_round_trip():
    pos = pd.Series([0, 1, 1, 0, 0.0])
    _, turnover, net = net_returns(pos, pd.Series(0.0, index=pos.index), beta=1.0, cost_bps=5)
    assert np.isclose(net.sum(), -2 * 5 / 1e4) and turnover.sum() == 2


def test_trade_pnl():
    pos = pd.Series([0, 1, 1, 0, 0.0])
    net = pd.Series([0, -0.0005, 0.01, 0.0015, 0.0])
    trades = extract_trades(pos, net)
    assert len(trades) == 1 and np.isclose(trades[0], 0.011)


def test_half_life():
    rng, s = np.random.default_rng(1), [0.0]
    for _ in range(5000):
        s.append(0.9 * s[-1] + rng.normal())
    assert 5 < half_life(pd.Series(s)) < 8.5   # true value is about 6.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
