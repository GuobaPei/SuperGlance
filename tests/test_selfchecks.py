"""The package's other CPU self-checks, run as their modules run them: the RL estimators of the ablation table (the
normalized estimators against the exact RL loss) and the AlpaSim driver (geometry, camera re-projection, past frames).

  python -m pytest -q tests
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(module, *args):
    out = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_rl_estimators():
    assert "selftest ok" in run("superglance.ablations.rl_estimators", "--selftest")


def test_alpasim_driver():
    assert "SELFTEST_OK" in run("superglance.alpasim.driver", "--selftest")
