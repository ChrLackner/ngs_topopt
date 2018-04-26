
import pytest

import sys
sys.path.insert(0,"../examples/timestepping")

def test_beam():
    import beam
    assert(beam.objective.GetValue() < 0.12)
    assert(abs(beam.constraints.values[0]) < 1e-2)

if __name__ == "__main__":
    test_beam()
