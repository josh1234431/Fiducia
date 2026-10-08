"""Functions a worker process must be able to import by name, for the tests."""

import time


def cache_setting(_):
    from rasterio.env import get_gdal_config

    return get_gdal_config("GDAL_CACHEMAX")


def hang_on_start(_cache_mb):
    """A worker initialiser that never finishes: a stuck pool."""
    time.sleep(3600)
