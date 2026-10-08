"""The Fiducia photogrammetry engine.

Module map:

===================  ==========================================================
``collinearity``     The collinearity condition and rotation conventions.
``camera``           Interior orientation: fiducials, distortion, chip geometry.
``resection``        Single-photo exterior orientation from control.
``bundle``           Block adjustment over many photos and tie points.
``geodesy``          Coordinate systems, including the South African LO zones.
``raster``           Tiled raster access, overviews, display rendering.
``project``          Autosave, journalling, snapshots, portable links.
``jobqueue``         Background job queue with progress and cancel.
``ortho``            Orthorectification by backward projection.
``mosaic``           Cutlines, colour balancing, blending.
``tiepoints``        Doomscroll — automatic tie point matching.
``stereo_dem``       Stereo DEM extraction from epipolar pairs.
``lidar``            LiDAR point clouds to elevation rasters.
``satellite_rpc``    Satellite RPC models and GCP refinement.
``reports``          Project and residual reports.
===================  ==========================================================
"""

__version__ = "0.2.0"
__all__ = [
    "bundle",
    "camera",
    "collinearity",
    "stereo_dem",
    "jobqueue",
    "geodesy",
    "mosaic",
    "satellite_rpc",
    "ortho",
    "project",
    "raster",
    "reports",
    "resection",
    "lidar",
    "tiepoints",
]
