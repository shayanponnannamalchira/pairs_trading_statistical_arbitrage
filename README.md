# Pairs-Trading Statistical Arbitrage (US Banks)

Scans a single-sector universe for cointegrated pairs, backtests a z-score mean-reversion
strategy out of sample with transaction costs, and prints today's BUY/SELL/HOLD signals.
An optional logistic-regression entry filter and an optional Alpaca paper-trading module are included.

## Run
```bash
pip install -r requirements.txt
python tests.py          # offline sanity checks
python main.py           # full pipeline -> outputs/
python main.py --live    # also print dry-run Alpaca orders
```
Needs internet access to Yahoo Finance on first run (prices are then cached in `data/prices.csv`).
Outputs: `outputs/REPORT.md` (tables + verdict), `results_*.csv`, `equity_curves.png`, `pair_*.png`.
Settings (universe, dates, thresholds, costs) are in `config.py`.

## Method
| Stage | Window | What happens |
|---|---|---|
| Train | 2019-01-01 to 2022-12-31 | Engle-Granger test on log prices for every pair; keep p < 0.05, beta > 0, half-life 2-60 days; OLS hedge ratio; rank by p-value + half-life rank; top 5. ML filter (if used) is fit here. |
| Test | 2023-01-01 to latest | Hedge ratios frozen. Trailing 60-day z-score. Long spread z < -2, short z > 2, exit \|z\| < 0.5, stop \|z\| > 4 or 40 days. Fill one day after the signal. 5 bps per side per leg. Forced exit on the last day. |

Signal meaning (beta = hedge ratio, Y/X pair):
- **BUY** = long the spread: buy $1 of Y, short $beta of X.
- **SELL** = short the spread: short $1 of Y, buy $beta of X.
- **HOLD** = flat (close any open position).

BUY/SELL reflect the *current target position*, so a pair already in a trade keeps showing BUY/SELL until it exits. The `event` column marks new entries/exits.

Metrics use daily net returns over every test day (flat days included). Portfolio = equal capital per pair. Sharpe uses rf = 0, 252 days.

## Testing status
`tests.py` checks that the z-score is causal, the entry/exit state machine, cost accounting, trade PnL and the half-life estimator.
The pipeline was smoke-tested end to end on *synthetic* prices only (plumbing check, not evidence of any edge).
**Real-data results have not been generated yet** - run `python main.py` and paste the numbers below.


> Out of sample (2023-01 to ___), the equal-weight portfolio returned __% net of 5 bps/side costs, Sharpe __, max drawdown __%; k of 5 pairs were profitable. The strategy **was / was not** profitable out of sample. Likely reason: ___
> ML filter vs none: Sharpe __ vs __, trades __ vs __. It **did / did not** help.

| Pair | Return | Sharpe | Max DD | Trades | Win rate | Turnover/yr |
|---|---|---|---|---|---|---|
| ... | | | | | | |
| PORTFOLIO | | | | | | |

## Limitations
- One train/test split, one test period. A good or bad result is a single draw, not proof.
- Multiple testing: with ~231 pairs, about 12 pass p < 0.05 by chance.
- Frozen hedge ratios: no re-estimation during the test window, so relationship drift hurts.
- Survivorship bias: the universe is today's large banks; failed/acquired banks are absent.
- Cost model: flat 5 bps per side. Ignores short borrow fees, market impact, spread widening in stress.
- Daily closes only, fills at the close, no intraday risk.
- The ML filter is trained on in-sample spreads, which look more mean-reverting than out-of-sample ones.
- Parameters (2.0 / 0.5 / 4.0, 60-day window) were set up front, not tuned on the test window. Changing them after seeing test results would invalidate it.

## Why the edge may not persist live
- Same-sector cointegration is regime-dependent; rate shocks and bank-specific events (e.g. 2023 regional-bank stress) can permanently break spreads.
- Stat-arb is crowded; well-known pairs get arbitraged away.
- Live frictions (borrow availability, partial fills, slippage in volatile markets) exceed backtest assumptions.
- Selection bias: pairs that look best in training are partly the luckiest ones.
- Stop-outs cluster in stress periods, when correlations spike and spreads diverge together.
