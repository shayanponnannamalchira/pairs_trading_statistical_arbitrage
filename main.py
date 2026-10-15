"""One command runs the whole pipeline: python main.py"""
import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest import backtest_pair, plot_equity, plot_pair, summarize
from config import Config
from data import load_prices
from ml_filter import fit_entry_filter
from pairs import select_pairs
from strategy import latest_signals


def pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["total_return"] = df["total_return"].map(pct)
    out["sharpe"] = df["sharpe"].round(2)
    out["max_drawdown"] = df["max_drawdown"].map(pct)
    out["trades"] = df["trades"].astype(int)
    out["win_rate"] = df["win_rate"].map(pct)
    out["turnover_per_yr"] = df["turnover_per_year"].round(1)
    return out


def verdict(table: pd.DataFrame, label: str, cfg: Config) -> str:
    p, rows = table.loc["PORTFOLIO"], table.drop("PORTFOLIO")
    word = "PROFITABLE" if p["total_return"] > 0 else "NOT profitable"
    return (f"{label}: {word} out of sample after {cfg.cost_bps} bps/side costs. Portfolio return "
            f"{pct(p['total_return'])}, Sharpe {p['sharpe']:.2f}, max drawdown {pct(p['max_drawdown'])}; "
            f"{int((rows['total_return'] > 0).sum())}/{len(rows)} pairs finished positive.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--no-ml", action="store_true", help="skip the ML filter")
    ap.add_argument("--live", action="store_true", help="print Alpaca orders (dry-run)")
    ap.add_argument("--execute", action="store_true", help="with --live: submit to PAPER account")
    args = ap.parse_args()

    cfg = Config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(exist_ok=True)

    prices = load_prices(cfg, refresh=args.refresh)
    train, test = prices.loc[:cfg.train_end], prices.loc[cfg.test_start:]
    assert train.index.max() < test.index.min(), "train and test windows overlap"
    window = (f"TRAIN {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} days) | "
              f"TEST {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)")
    print(window)

    # ---- pair selection sees TRAIN ONLY ----
    pairs, info = select_pairs(train, cfg)
    print(f"Tested {info['n_tests']} pairs; {info['n_coint']} had p<{cfg.pvalue_max} "
          f"(~{info['expected_false_positives']} expected by chance alone); "
          f"{info['n_after_filters']} passed beta/half-life filters.")
    if not pairs:
        print("No pairs selected. Nothing to backtest.")
        return
    pairs_df = pd.DataFrame([asdict(p) for p in pairs]).round(4)
    print("\nSelected pairs (train window):\n", pairs_df.to_string(index=False))

    # ---- out-of-sample backtest, all selected pairs reported ----
    base = [backtest_pair(prices, p, cfg) for p in pairs]
    base_tbl = summarize(base)

    ml_res, ml_tbl, ml_note, filt = None, None, "ML filter skipped.", None
    if not args.no_ml:
        filt = fit_entry_filter(train, pairs, cfg)
        if filt is None:
            ml_note = "ML filter: too few training events or a single class; not used."
        else:
            ml_res = [backtest_pair(prices, p, cfg, filt.entry_ok(prices, p, cfg)) for p in pairs]
            ml_tbl = summarize(ml_res)
            b, m = base_tbl.loc["PORTFOLIO"], ml_tbl.loc["PORTFOLIO"]
            ml_note = (f"ML filter (logistic regression, {filt.n_train} train events, base rate "
                       f"{filt.base_rate:.2f}, threshold {filt.threshold:.2f}). Test-window change vs no filter: "
                       f"return {pct(m['total_return'] - b['total_return'])} pts, "
                       f"Sharpe {m['sharpe'] - b['sharpe']:+.2f}, trades {int(b['trades'])} -> {int(m['trades'])}. "
                       f"{'The filter helped on this window.' if m['sharpe'] > b['sharpe'] else 'The filter did NOT help on this window.'}")

    # ---- outputs ----
    base_tbl.to_csv(out_dir / "results_baseline.csv")
    if ml_tbl is not None:
        ml_tbl.to_csv(out_dir / "results_ml_filter.csv")
    plot_equity(base, ml_res, str(out_dir / "equity_curves.png"))
    for r in base:
        plot_pair(r, cfg, str(out_dir / f"pair_{r.pair.y}_{r.pair.x}.png"))

    signals = latest_signals(prices, pairs, cfg)
    lines = [f"# Results\n\n{window}\n\nCosts: {cfg.cost_bps} bps per side per leg, "
             f"signals filled {cfg.exec_lag} day(s) after the close. All numbers are net of costs.\n",
             f"## Verdict\n\n{verdict(base_tbl, 'Baseline', cfg)}\n",
             f"## Selected pairs (training window)\n\n```\n{pairs_df.to_string(index=False)}\n```\n",
             f"## Baseline, out of sample (all selected pairs)\n\n```\n{fmt(base_tbl).to_string()}\n```\n"]
    if ml_tbl is not None:
        lines += [f"## With ML filter\n\n```\n{fmt(ml_tbl).to_string()}\n```\n"]
    lines += [f"## ML comparison\n\n{ml_note}\n",
              f"## Latest signals (as of last close; execute next session)\n\n```\n{signals.to_string(index=False)}\n```\n"]
    (out_dir / "REPORT.md").write_text("\n".join(lines))

    print("\n" + verdict(base_tbl, "Baseline", cfg))
    print("\nBaseline results (out of sample, net of costs):\n", fmt(base_tbl).to_string())
    if ml_tbl is not None:
        print("\nWith ML filter:\n", fmt(ml_tbl).to_string())
    print("\n" + ml_note)
    print("\nToday's signals:\n", signals.to_string(index=False))
    print(f"\nSaved report, CSVs and charts to {out_dir}/")

    if args.live:
        from live import print_orders
        print_orders(prices, signals, pairs, cfg, execute=args.execute)


if __name__ == "__main__":
    main()
