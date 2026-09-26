"""
Camera projection module for VISIONCTRL.

Maps MediaPipe normalized landmark coordinates and 3D spatial rays to 2D image pixel coordinates.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from pointing_estimator import SpatialRay


@dataclass
class ProjectedRay:
    """2D projected spatial ray in camera image pixel coordinates."""
    origin_px: Tuple[int, int]
    endpoint_px: Tuple[int, int]
    visible: bool


class CameraProjection:
    """Projects normalized landmark coordinates and 3D rays to image pixels."""

    @staticmethod
    def normalized_to_pixel(
        nx: float,
        ny: float,
        frame_width: int,
        frame_height: int,
        clamp: bool = False,
    ) -> Tuple[int, int]:
        """
        Maps normalized coordinates [0.0, 1.0] to image pixels.

        Args:
            nx: Normalized X coordinate.
            ny: Normalized Y coordinate.
            frame_width: Width of image in pixels.
            frame_height: Height of image in pixels.
            clamp: If True, clamps output within [0, dim-1].

        Returns:
            Tuple of (px, py) integer pixel coordinates.
        """
        px = int(round(nx * frame_width))
        py = int(round(ny * frame_height))

        if clamp:
            px = max(0, min(frame_width - 1, px))
            py = max(0, min(frame_height - 1, py))

        return (px, py)

    @classmethod
    def project_ray(
        cls,
        ray: SpatialRay,
        frame_width: int,
        frame_height: int,
        visual_length_px: Optional[float] = None,
    ) -> ProjectedRay:
        """
        Projects a 3D SpatialRay onto the 2D image plane.

        Args:
            ray: 3D SpatialRay in landmark coordinate space.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            visual_length_px: Optional visual length in pixels.

        Returns:
            ProjectedRay containing 2D origin and target pixels.
        """
        orig_x, orig_y = cls.normalized_to_pixel(
            ray.origin_3d[0],
            ray.origin_3d[1],
            frame_width,
            frame_height,
            clamp=False,
        )

        if visual_length_px is not None and visual_length_px > 0:
            dir_px_x = ray.direction_3d[0] * frame_width
            dir_px_y = ray.direction_3d[1] * frame_height
            mag_2d = math.sqrt(dir_px_x * dir_px_x + dir_px_y * dir_px_y)

            if mag_2d > 1e-6:
                norm_dir_x = dir_px_x / mag_2d
                norm_dir_y = dir_px_y / mag_2d
                end_x = int(round(orig_x + norm_dir_x * visual_length_px))
                end_y = int(round(orig_y + norm_dir_y * visual_length_px))
            else:
                end_x, end_y = cls.normalized_to_pixel(
                    ray.endpoint_3d[0],
                    ray.endpoint_3d[1],
                    frame_width,
                    frame_height,
                    clamp=False,
                )
        else:
            end_x, end_y = cls.normalized_to_pixel(
                ray.endpoint_3d[0],
                ray.endpoint_3d[1],
                frame_width,
                frame_height,
                clamp=False,
            )

        return ProjectedRay(
            origin_px=(orig_x, orig_y),
            endpoint_px=(end_x, end_y),
            visible=ray.valid,
        )
