#!/usr/bin/env python3
"""Generate schema reports from a CSV or a DataFrame fixture.

Usage:
  python scripts/generate_report.py --csv data/EURUSD_Daily.csv --pair EURUSD
  python scripts/generate_report.py --fixture data/tests/fundamentals/sample_fundamentals_series.json --pair EURUSD
"""
import argparse
import os
import json
import pandas as pd

from .report_utils import generate_schema_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='Path to CSV file to generate report from')
    parser.add_argument('--fixture', help='Path to JSON fixture')
    parser.add_argument('--pair', required=True, help='Pair name for report')
    parser.add_argument('--out', default='output', help='Output directory')
    args = parser.parse_args()

    df = None
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.fixture:
        with open(args.fixture, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    report = generate_schema_report(df, pair=args.pair, out_dir=args.out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
