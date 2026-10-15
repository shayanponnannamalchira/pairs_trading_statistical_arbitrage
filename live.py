"""Optional paper trading via Alpaca. DRY-RUN unless execute=True. Keys come from env vars only."""
import os

import pandas as pd

from config import Config
from pairs import Pair


def target_shares(signals: pd.DataFrame, pairs: list[Pair], prices: pd.DataFrame, cfg: Config) -> dict:
    per_pair = cfg.live_capital_usd / len(pairs)
    targets: dict = {}
    for p in pairs:
        row = signals[signals["pair"] == f"{p.y}/{p.x}"].iloc[0]
        direction = {"BUY": 1, "SELL": -1, "HOLD": 0}[row["signal"]]
        y_usd = per_pair / (1 + abs(p.beta))
        x_usd = abs(p.beta) * y_usd
        y_sh = int(y_usd // prices[p.y].iloc[-1])
        x_sh = int(x_usd // prices[p.x].iloc[-1])
        targets[p.y] = targets.get(p.y, 0) + direction * y_sh   # pairs can share a ticker: net them
        targets[p.x] = targets.get(p.x, 0) - direction * x_sh
    return targets


def print_orders(prices, signals, pairs, cfg: Config, execute: bool = False) -> None:
    targets = target_shares(signals, pairs, prices, cfg)
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    client, current = None, {}
    if key and secret:
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(key, secret, paper=True)
            current = {pos.symbol: int(float(pos.qty)) for pos in client.get_all_positions()}
        except ImportError:
            print("alpaca-py not installed: assuming a flat account, dry-run only.")
    else:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set: assuming a flat account, dry-run only.")

    print(f"\n{'EXECUTING on paper account' if execute and client else 'DRY RUN - no orders sent'}")
    for sym in sorted(set(targets) | set(current)):
        delta = targets.get(sym, 0) - current.get(sym, 0)
        if delta == 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        print(f"  {side:4s} {abs(delta):5d} {sym}  (target {targets.get(sym, 0):+d}, current {current.get(sym, 0):+d})")
        if execute and client:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
            client.submit_order(MarketOrderRequest(
                symbol=sym, qty=abs(delta),
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                time_in_force=TimeInForce.DAY))
