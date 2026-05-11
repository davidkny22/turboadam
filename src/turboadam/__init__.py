from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("turboadam")
except PackageNotFoundError:
    __version__ = "unknown"

from turboadam.optimizer import TurboAdam

__all__ = ["TurboAdam"]
