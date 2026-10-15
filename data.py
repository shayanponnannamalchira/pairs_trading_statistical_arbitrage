"""Download, cache and clean adjusted close prices."""
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import Config


def _download(cfg: Config) -> pd.DataFrame:
    raw = yf.download(cfg.universe, start=cfg.start, end=cfg.end,
                      auto_adjust=True, progress=False)
    return raw["Close"]


def _cache_is_usable(cached: pd.DataFrame, cfg: Config) -> bool:
    target_end = pd.Timestamp(cfg.end) if cfg.end else pd.Timestamp.today().normalize()
    fresh = cached.index.max() >= target_end - pd.Timedelta(days=5)
    covers_start = cached.index.min() <= pd.Timestamp(cfg.start) + pd.Timedelta(days=7)
    covers_universe = set(cfg.universe) <= set(cached.columns)
    return fresh and covers_start and covers_universe


def clean_prices(prices: pd.DataFrame, cfg: Config, max_missing: float = 0.05) -> pd.DataFrame:
    prices = prices.sort_index().loc[cfg.start:cfg.end]
    prices = prices[[c for c in cfg.universe if c in prices.columns]]
    prices = prices.dropna(axis=1, how="all")
    sparse = prices.columns[prices.isna().mean() > max_missing]
    if len(sparse):
        print(f"Dropping tickers with >{max_missing:.0%} missing data: {list(sparse)}")
        prices = prices.drop(columns=sparse)
    prices = prices.ffill(limit=3).dropna()  # fill short gaps, drop rows still incomplete
    return prices[(prices > 0).all(axis=1)]


def load_prices(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    cache = Path(cfg.data_dir) / "prices.csv"
    prices = None
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if _cache_is_usable(cached, cfg):
            prices = cached
    if prices is None:
        print("Downloading prices from yfinance...")
        prices = _download(cfg).dropna(axis=1, how="all")
        if prices.empty or prices.shape[1] < 2:
            raise RuntimeError(
                "yfinance returned no usable price data (check your internet connection / "
                "Yahoo availability, then re-run). Nothing was cached.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return clean_prices(prices, cfg)
