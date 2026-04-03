"""
cvdatakit – Dataset-centric CV toolkit.

Modules
-------
stats       : Dataset statistics and class-imbalance metrics.
quality     : Label-quality checks and mislabel detection.
active_learning : Active-learning strategies and loop orchestration.
recipes     : Ready-made dataset configurations (COCO, ImageNet …).
io          : COCO-format reader and HTML/JSON report generator.
cli         : Command-line interface (``cvdatakit`` command).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cvdatakit")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = ["__version__"]
