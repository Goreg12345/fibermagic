from .core.demodulate import demodulate
from .IO.npm import read_npm, merge_events
from .core.perievents import perievents
from .utils.download_dataset import download
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
