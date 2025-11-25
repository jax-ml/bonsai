import pytest
from clip.run_model import run_demo

def test_run_demo_smoke():
    # smoke test that demo runs without raising
    run_demo()
