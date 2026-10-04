"""stno: macrospin simulation of dipolar-coupled spin-torque nano-oscillators (STNOs)."""
from .constants import POLARITY_ANTIPARALLEL, POLARITY_PARALLEL
from .params import default_params
from .dynamics import simulate

__version__ = "0.1.0"
__all__ = ["simulate", "default_params", "POLARITY_ANTIPARALLEL", "POLARITY_PARALLEL"]
