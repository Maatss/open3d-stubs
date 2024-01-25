"""
This type stub file was generated by pyright.
"""

from typing import Callable

def make_dir(folder_name):  # -> None:
    """Create a directory.

    If already exists, do nothing
    """
    ...

def get_hash(x: str):  # -> str:
    """Generate a hash from a string."""
    ...

class Cache:
    """Cache converter for preprocessed data."""

    def __init__(self, func: Callable, cache_dir: str, cache_key: str) -> None:
        """Initialize.

        Args:
            func: preprocess function of a model.
            cache_dir: directory to store the cache.
            cache_key: key of this cache
        Returns:
            class: The corresponding class.
        """
        ...
    def __call__(self, unique_id: str, *data):
        """Call the converter. If the cache exists, load and return the cache,
        otherwise run the preprocess function and store the cache.

        Args:
            unique_id: A unique key of this data.
            data: Input to the preprocess function.

        Returns:
            class: Preprocessed (cache) data.
        """
        ...
