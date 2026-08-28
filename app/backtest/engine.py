"""Core backtesting engine, with checkpoint/resume support.

Checkpointing idea:
  The walk-forward loop already processes one fold at a time. After each fold
  completes, the caller (jobs.py) persists a "checkpoint" -- the list of
  completed folds, each carrying its own out-of-sample returns -- to the job
  row in Postgres. If the worker process dies mid-run and the job is retried,
  `walk_forward` is handed that checkpoint, skips recomputing the folds
  already in it, and continues from the next one. This turns a crash mid-way
  through a 20-fold run into "redo the last unfinished fold," not "start over."

Correctness note: a fold's optimize+evaluate step is deterministic given the
same data and params, so re-doing it on resume (if no checkpoint were kept)
would be *safe* but wasteful for expensive grids/long histories -- that
wasted recompute is exactly what checkpointing avoids.
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
    position = signal.shift(1).fillna(0.0)  # LOOKAHEAD PREVENTION

    asset_ret = df["close"].pct_change().fillna(0.0)
    gross = position * asset_ret

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


def _fold_windows(n: int, n_splits: int) -> list[tuple[int, int]]:
    """Compute the (start_test, test_size) boundaries once, so checkpointed
    and fresh runs agree on exactly where each fold begins."""
    min_train = max(60, n // (n_splits + 1))
    test_size = max(20, (n - min_train) // n_splits)
    windows = []
    start_test = min_train
    fold = 0
    while start_test + test_size <= n and fold < n_splits:
        windows.append((start_test, test_size))
        start_test += test_size
        fold += 1
    return windows


def _serialize_fold_returns(r: pd.Series) -> list[dict]:
    return [{"date": str(ts.date()), "ret": float(v)} for ts, v in r.items()]


def _deserialize_fold_returns(rows: list[dict]) -> pd.Series:
    idx = pd.to_datetime([row["date"] for row in rows], utc=True)
    vals = [row["ret"] for row in rows]
    return pd.Series(vals, index=idx, name="ret")


def walk_forward(
    df: pd.DataFrame,
    strategy: str,
    commission_bps: float,
    slippage_bps: float,
    n_splits: int = 5,
    progress_cb: ProgressCB = None,
    checkpoint: Optional[list[dict]] = None,
    checkpoint_cb: Optional[Callable[[list[dict]], None]] = None,
) -> tuple[pd.Series, list[dict]]:
    """Walk-forward validation, resumable from a checkpoint.

    checkpoint: previously completed folds, e.g. from a prior (crashed) run:
        [{"fold": 1, "params": {...}, "train_sharpe": .., "test_sharpe": ..,
          "test_start": .., "test_end": .., "test_bars": ..,
          "oos_returns": [{"date": .., "ret": ..}, ...]}, ...]
    checkpoint_cb: called with the full up-to-date fold list after EVERY fold
        (including ones restored from checkpoint on the first call), so the
        caller can persist it. Kept synchronous and cheap: just a DB write.
    """
    n = len(df)
    if n < 80:
        raise ValueError(f"not enough bars to backtest ({n}); need >= 80")

    windows = _fold_windows(n, n_splits)
    if not windows:
        raise ValueError("not enough bars to form a single walk-forward fold")

    folds: list[dict] = list(checkpoint) if checkpoint else []
    oos_chunks: list[pd.Series] = [
        _deserialize_fold_returns(f["oos_returns"]) for f in folds
    ]
    already_done = len(folds)

    if already_done:
        if progress_cb:
            progress_cb(already_done, len(windows), f"Resumed from checkpoint ({already_done} folds already done)")
        if checkpoint_cb:
            checkpoint_cb(folds)  # let caller confirm/persist the restored state

    for fold_idx in range(already_done, len(windows)):
        start_test, test_size = windows[fold_idx]
        train = df.iloc[:start_test]
        test = df.iloc[start_test : start_test + test_size]

        best_params, train_sharpe = _optimize(train, strategy, commission_bps, slippage_bps)

        warmup = max_lookback(strategy, best_params)
        eval_slice = df.iloc[max(0, start_test - warmup) : start_test + test_size]
        r_full = strategy_returns(eval_slice, strategy, best_params, commission_bps, slippage_bps)
        r_test = r_full.loc[test.index]
        oos_chunks.append(r_test)

        fold_record = {
            "fold": fold_idx + 1,
            "params": best_params,
            "train_sharpe": round(train_sharpe, 4),
            "test_sharpe": round(metrics.sharpe(r_test), 4),
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "test_bars": int(len(test)),
            "oos_returns": _serialize_fold_returns(r_test),
        }
        folds.append(fold_record)

        if checkpoint_cb:
            checkpoint_cb(folds)  # persist progress after EVERY fold, not just at the end
        if progress_cb:
            progress_cb(fold_idx + 1, len(windows), f"Completed fold {fold_idx + 1}/{len(windows)}")

    oos_returns = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float, name="ret")
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
    checkpoint: Optional[list[dict]] = None,
    checkpoint_cb: Optional[Callable[[list[dict]], None]] = None,
) -> dict:
    if progress_cb:
        progress_cb(0, n_splits, "Starting walk-forward validation")

    oos_returns, folds = walk_forward(
        df, strategy, commission_bps, slippage_bps, n_splits,
        progress_cb=progress_cb, checkpoint=checkpoint, checkpoint_cb=checkpoint_cb,
    )

    if progress_cb:
        progress_cb(len(folds), len(folds), "Computing metrics + bootstrap")

    eq = metrics.equity_curve(oos_returns, initial_cash)
    pval = metrics.bootstrap_sharpe_pvalue(oos_returns)

    # Strip the heavy per-fold oos_returns before returning the final result;
    # they were only needed for checkpointing/resuming, not for the report.
    folds_out = [{k: v for k, v in f.items() if k != "oos_returns"} for f in folds]

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
        "folds": folds_out,
        "equity_curve": [
            {"date": str(ts.date()), "equity": round(float(v), 2)} for ts, v in eq.items()
        ],
    }
    return result
