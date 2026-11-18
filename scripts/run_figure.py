#!/usr/bin/env python3
"""Run a selected original figure, with its relative source tables resolved."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    entries = json.loads((ROOT / 'docs/figure_entries.json').read_text())['entries']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=entries)
    args = parser.parse_args()
    directory = ROOT / 'figures'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(directory)
    sys.path[:0] = [str(directory), str(ROOT)]
    sys.argv = [str(directory / (args.figure + '.py'))]
    runpy.run_path(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    main()
