
import pytest

import sys
sys.path.insert(0,"../examples/timestepping")

def test_wave():
    import wave
    assert(wave.objective.GetValue() < -1.5)

if __name__ == "__main__":
    test_wave()
