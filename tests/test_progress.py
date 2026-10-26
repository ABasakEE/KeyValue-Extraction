"""Unit tests for FlushProgressBar in-place rendering."""

import io
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.progress import FlushProgressBar, tqdm


def test_progress_bar_basic_iteration():
    buf = io.StringIO()
    data = list(range(10))
    pbar = FlushProgressBar(data, desc="Epoch 1", leave=False, ncols=80, file=buf, mininterval=0.0)
    
    collected = []
    for item in pbar:
        collected.append(item)
        pbar.set_postfix(loss="0.500")

    assert collected == data
    assert pbar.n == 10
    output = buf.getvalue()
    # Output must contain carriage returns, description, and leave=False blanking
    assert "\r" in output
    assert "Epoch 1" in output
    assert "10/10" in output
    # Since leave=False, it should erase at the end
    assert output.endswith("\r" + " " * 80 + "\r")


def test_progress_bar_leave_true():
    buf = io.StringIO()
    pbar = FlushProgressBar(range(5), desc="Phase 1", leave=True, ncols=80, file=buf, mininterval=0.0)
    for _ in pbar:
        pass
    output = buf.getvalue()
    assert output.endswith("\n")


def test_progress_bar_manual_update():
    buf = io.StringIO()
    with FlushProgressBar(total=20, desc="Manual", ncols=80, file=buf, mininterval=0.0) as pbar:
        assert len(pbar) == 20
        pbar.update(10)
        assert pbar.n == 10
        pbar.update(10)
        assert pbar.n == 20


def test_tqdm_alias():
    buf = io.StringIO()
    pbar = tqdm(range(3), file=buf, mininterval=0.0)
    items = list(pbar)
    assert items == [0, 1, 2]
    assert pbar.n == 3


def test_disabled_progress_bar():
    buf = io.StringIO()
    pbar = FlushProgressBar(range(5), disable=True, file=buf)
    items = list(pbar)
    assert items == [0, 1, 2, 3, 4]
    assert buf.getvalue() == ""


if __name__ == "__main__":
    test_progress_bar_basic_iteration()
    test_progress_bar_leave_true()
    test_progress_bar_manual_update()
    test_tqdm_alias()
    test_disabled_progress_bar()
    print("[PASS] All FlushProgressBar tests passed successfully!")
