"""NVSonar - Active GPU diagnostic tool"""

from importlib.metadata import PackageNotFoundError, version

from nvsonar.session import monitor, print_summary, start, stop

try:
    __version__ = version("nvsonar")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "monitor", "print_summary", "start", "stop"]
