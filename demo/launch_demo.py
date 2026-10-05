"""One-click launcher; installs missing dependencies in a private environment."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10 or newer is required.')
    interpreter = Path(sys.executable)
    if any(importlib.util.find_spec(name) is None for name in ['numpy', 'scipy']):
        environment = root / '.venv'
        interpreter = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not interpreter.exists():
            print('Preparing a private Python environment for the demo...', flush=True)
            subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
        dependencies = subprocess.run([str(interpreter), '-c', 'import numpy, scipy'],
                                      capture_output=True)
        if dependencies.returncode:
            print('Installing NumPy and SciPy (first run only)...', flush=True)
            subprocess.run([str(interpreter), '-m', 'pip', 'install', '-r', str(root / 'requirements.txt')], check=True)
    subprocess.run([str(interpreter), str(root / 'run_demo.py'), *sys.argv[1:]], check=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Demo could not complete: {error}', file=sys.stderr)
        sys.exit(1)
