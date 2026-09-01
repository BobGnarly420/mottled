import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# tests/ too, so shared fixtures (tests/tiny.py) import by plain name from any
# test module regardless of which directory pytest was invoked from.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
