import os
from clip.run_model import run_demo

PAPER_PATH = "/mnt/data/2103.00020v1.pdf"  # local path to the uploaded paper

def test_paper_exists():
    assert os.path.exists(PAPER_PATH), f"Paper must be present at {PAPER_PATH}"

def test_demo_runs():
    run_demo()  # should not raise
