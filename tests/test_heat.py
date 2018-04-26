
import pytest

import sys
sys.path.insert(0,"../examples/timestepping")

def test_heat():
    import heat
    assert(heat.objective.GetValue() < -0.04)
    assert(abs(heat.constr.values[0]) < 2e-2)

if __name__ == "__main__":
    test_heat()
