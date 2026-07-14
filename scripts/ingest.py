"""CLI: ingest historical bars for one or more tickers.

Usage:
    python -m scripts.ingest AAPL MSFT SPY --start 2018-01-01 --end 2024-12-31
"""
import argparse
import asyncio
import datetime as dt
import logging

from app.ingestion import ingest_tickers


def _date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest OHLCV bars into PostgreSQL")
    parser.add_argument("tickers", nargs="+", help="ticker symbols, e.g. AAPL MSFT SPY")
    parser.add_argument("--start", type=_date, default=_date("2018-01-01"))
    parser.add_argument("--end", type=_date, default=dt.date.today())
    parser.add_argument("--interval", default="1d", help="1d, 1h, etc.")
    args = parser.parse_args()

    summary = asyncio.run(
        ingest_tickers([t.upper() for t in args.tickers], args.start, args.end, args.interval)
    )
    print(f"Ingested {summary['total_bars']} bars")
    for r in summary["per_ticker"]:
        print(f"  {r['ticker']}: {r['bars']} bars, {r['gaps']} gaps")


if __name__ == "__main__":
    main()
