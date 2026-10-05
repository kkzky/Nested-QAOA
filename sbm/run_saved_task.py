"""Run a saved SBM configuration with a user-selected output location."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True, help='Zero-based task index')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.config.read_text(encoding='utf-8'))
    if not 0 <= args.index < len(tasks):
        parser.error('--index is outside the saved task list')
    task = tasks[args.index]
    runner = Path(__file__).resolve().parent / 'code' / Path(task['runner']).name
    command = [sys.executable, str(runner), '--output', str(args.output)]
    for key, value in task['args'].items():
        if key == 'output' or value is None:
            continue
        if key == 'lp_configs':
            value = ','.join(':'.join(map(str, pair)) for pair in value)
        elif isinstance(value, list):
            value = ','.join(map(str, value))
        command += ['--' + key.replace('_', '-'), str(value)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
