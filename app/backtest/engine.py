"""Core backtesting engine.

Key correctness guarantees:
  * Lookahead prevention: a signal computed using data up to bar t is only
    *acted on* at bar t+1 (positions are shifted forward by one bar).
  * Costs: commission + slippage charged on turnover (change in position).
  * Walk-forward: parameters are optimized on an expanding in-sample window and
    evaluated only on the subsequent, untouched out-of-sample window. Reported
    metrics use the concatenated out-of-sample returns only.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from . import metrics
from .strategies import compute_signal, max_lookback, param_grid

ProgressCB = Optional[Callable[[int, int, str], None]]


def strategy_returns(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    commission_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """Lookahead-safe net daily returns for a strategy over `df`."""
    signal = compute_signal(df, strategy, params)
    # LOOKAHEAD PREVENTION: trade on the *next* bar after the signal forms.
    position = signal.shift(1).fillna(0.0)

    asset_ret = df["close"].pct_change().fillna(0.0)
    gross = position * asset_ret

    # Turnover = |change in position|; first bar's entry counts as turnover.
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0]) if len(position) else 0.0
    cost_rate = (commission_bps + slippage_bps) / 10_000.0
    costs = turnover * cost_rate

    return (gross - costs).rename("ret")


def _optimize(
    train: pd.DataFrame, strategy: str, commission_bps: float, slippage_bps: float
) -> tuple[dict, float]:
    best_params, best_sharpe = None, float("-inf")
    for params in param_grid(strategy):
        r = strategy_returns(train, strategy, params, commission_bps, slippage_bps)
        sh = metrics.sharpe(r)
        if sh > best_sharpe:
            best_sharpe, best_params = sh, params
    return best_params, best_sharpe


def walk_forward(
    df: pd.DataFrame,
    strategy: str,
    commission_bps: float,
    slippage_bps: float,
    n_splits: int = 5,
    progress_cb: ProgressCB = None,
) -> tuple[pd.Series, list[dict]]:
    n = len(df)
    if n < 80:
        raise ValueError(f"not enough bars to backtest ({n}); need >= 80")

    min_train = max(60, n // (n_splits + 1))
    test_size = max(20, (n - min_train) // n_splits)

    oos_chunks: list[pd.Series] = []
    folds: list[dict] = []

    start_test = min_train
    fold = 0
    while start_test + test_size <= n and fold < n_splits:
        train = df.iloc[:start_test]  # expanding in-sample window
        test = df.iloc[start_test : start_test + test_size]

        best_params, train_sharpe = _optimize(train, strategy, commission_bps, slippage_bps)

        # Include warmup bars from before the test window so indicators are valid
        # at the test window's first bar (warmup uses past data only -> no leak).
        warmup = max_lookback(strategy, best_params)
        eval_slice = df.iloc[max(0, start_test - warmup) : start_test + test_size]
        r_full = strategy_returns(eval_slice, strategy, best_params, commission_bps, slippage_bps)
        r_test = r_full.loc[test.index]
        oos_chunks.append(r_test)

        folds.append(
            {
                "fold": fold + 1,
                "params": best_params,
                "train_sharpe": round(train_sharpe, 4),
                "test_sharpe": round(metrics.sharpe(r_test), 4),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "test_bars": int(len(test)),
            }
        )

        fold += 1
        start_test += test_size
        if progress_cb:
            progress_cb(fold, n_splits, f"Completed fold {fold}/{n_splits}")

    oos_returns = (
        pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float, name="ret")
    )
    return oos_returns, folds


def run_backtest(
    df: pd.DataFrame,
    *,
    strategy: str,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
    n_splits: int,
    progress_cb: ProgressCB = None,
) -> dict:
    if progress_cb:
        progress_cb(0, n_splits, "Starting walk-forward validation")

    oos_returns, folds = walk_forward(
        df, strategy, commission_bps, slippage_bps, n_splits, progress_cb
    )

    if progress_cb:
        progress_cb(n_splits, n_splits, "Computing metrics + bootstrap")

    eq = metrics.equity_curve(oos_returns, initial_cash)
    pval = metrics.bootstrap_sharpe_pvalue(oos_returns)

    result = {
        "metrics": {
            "oos_sharpe": round(metrics.sharpe(oos_returns), 4),
            "max_drawdown": round(metrics.max_drawdown(eq), 4),
            "cagr": round(metrics.cagr(eq), 4),
            "total_return": round(float(eq.iloc[-1] / eq.iloc[0] - 1.0), 4) if len(eq) else 0.0,
            "n_oos_days": int(len(oos_returns)),
            "sharpe_pvalue": round(pval, 4),
            "significant_at_5pct": bool(pval < 0.05),
        },
        "folds": folds,
        "equity_curve": [
            {"date": str(ts.date()), "equity": round(float(v), 2)}
            for ts, v in eq.items()
        ],
    }
    return result
