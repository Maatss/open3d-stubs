from enum import Enum
from math import inf
from typing import Any, Iterable, Iterator, Optional, Sequence, Union, overload
from __future__ import annotations

from .. import core, geometry

class OrientedBoundingBox:
    @property
    def center(self) -> core.Tensor:
        """Returns the center for box."""
        ...
    @property
    def color(self) -> core.Tensor:
        """Returns the color for box."""
        ...
    @property
    def device(self) -> core.Device:
        """Returns the device of the geometry."""
        ...
    @property
    def dtype(self) -> core.Dtype:
        """Returns the data type attribute of this OrientedBoundingBox."""
        ...
    @property
    def extent(self) -> core.Tensor:
        """Returns the extent for box coordinates."""
        ...
    @property
    def is_cpu(self) -> bool:
        """Returns true if the geometry is on CPU."""
        ...
    @property
    def is_cuda(self) -> bool:
        """Returns true if the geometry is on CUDA."""
        ...
    @property
    def material(self) -> Any: # todo
        ...
    @property
    def rotation(self) -> core.Tensor:
        """Returns the rotation for box."""
        ...

    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> OrientedBoundingBox:
        """Clear all elements in the geometry."""
        ...
    def clone(self) -> OrientedBoundingBox:
        """Returns copy of the oriented box on the same device."""
        ...
    def cpu(self) -> OrientedBoundingBox:
        """Transfer the oriented box to CPU. If the oriented box is already on CPU, no copy will be performed."""
        ...
    @staticmethod
    def create_from_axis_aligned_bounding_box(aabb: AxisAlignedBoundingBox) -> OrientedBoundingBox:
        """Create an OrientedBoundingBox from the AxisAlignedBoundingBox.
        
        Args:
            aabb (AxisAlignedBoundingBox): AxisAlignedBoundingBox object from which OrientedBoundingBox is created.
        
        Returns:
            OrientedBoundingBox"""
        ...
    @staticmethod
    def create_from_points(points: core.Tensor, robust: bool = False) -> OrientedBoundingBox:
        """Creates an oriented bounding box using a PCA. Note that this is only an approximation to the minimum oriented bounding box that could be computed for example with O’Rourke’s algorithm (cf. http://cs.smith.edu/~jorourke/Papers/MinVolBox.pdf, https://www.geometrictools.com/Documentation/MinimumVolumeBox.pdf) This is a wrapper for a CPU implementation.
        
        Args:
            points (core.Tensor): A list of points with data type of float32 or float64 (N x 3 tensor, where N must be larger than 3).
            robust (bool, optional): If set to true uses a more robust method which works in degenerate cases but introduces noise to the points coordinates. Defaults to False.

        Returns:
            OrientedBoundingBox"""
        ...
    def cuda(self, device_id: int = 0) -> OrientedBoundingBox:
        """Transfer the oriented box to a CUDA device. If the oriented box is already on the specified CUDA device, no copy will be performed.
        
        Args:
            device_id (int, optional): Defaults to 0.
        
        Returns:
            OrientedBoundingBox"""
        ...
    @staticmethod
    def from_legacy(
        box: geometry.OrientedBoundingBox,
        dtype: core.Dtype = core.Dtype.Float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> OrientedBoundingBox:
        """Create an OrientedBoundingBox from a legacy Open3D oriented box.
        
        Args:
            box (geometry.OrientedBoundingBox):
            dtype (core.Dtype, optional): Defaults to core.Dtype.Float32.
            device (core.Device, optional): Defaults to core.Device("CPU:0").
            
        Returns:
            OrientedBoundingBox"""
        ...
    def get_axis_aligned_bounding_box(self) -> AxisAlignedBoundingBox:
        """Returns an oriented bounding box from the AxisAlignedBoundingBox."""
        ...
    def get_box_points(self) -> core.Tensor:
        """Returns the eight points that define the bounding box. The Return tensor has shape {8, 3} and data type same as the box."""
        ...
    def get_max_bound(self) -> core.Tensor:
        """Returns the max bound for box."""
        ...
    def get_min_bound(self) -> core.Tensor:
        """Returns the min bound for box."""
        ...
    def get_point_indices_within_bounding_box(points: core.Tensor) -> core.Tensor:
        """Indices to points that are within the bounding box.
        
        Args:
            points (core.Tensor): Tensor with {N, 3} shape, and type float32 or float64.
            
        Returns:
            core.Tensor"""
        ...
    def has_valid_material(self) -> bool:
        """Returns true if the geometry's material is valid."""
        ...
    def is_empty(self) -> bool:
        """Returns True iff the geometry is empty."""
        ...
    def rotate(self, rotation: core.Tensor, center: Optional[core.Tensor] = None) -> OrientedBoundingBox:
        """Rotate the oriented box by the given rotation matrix. If the rotation matrix is not orthogonal, the rotation will no be applied. The rotation center will be the box center if it is not specified.
        
        Args:
            rotation (core.Tensor): Rotation matrix of shape {3, 3}, type float32 or float64, device same as the box.
            center (Optional[core.Tensor], optional): Center of the rotation, default is null, which means use center of the box as rotation center.
        
        Returns:
            OrientedBoundingBox"""
        ...
    def scale(self, scale: float, center: Optional[core.Tensor] = None) -> OrientedBoundingBox:
        """Scale the axis-aligned box. If f$mif$ is the min_bound and f$maf$ is the max_bound of the axis aligned bounding box, and f$sf$ and f$cf$ are the provided scaling factor and center respectively, then the new min_bound and max_bound are given by f$mi = c + s (mi - c)f$ and f$ma = c + s (ma - c)f$. The scaling center will be the box center if it is not specified.
        
        Args:
            scale (float): The scale parameter.
            center (Optional[core.Tensor], optional): Center used for the scaling operation. Tensor with {3,} shape, and type float32 or float64
        
        Returns:
            OrientedBoundingBox"""
        ...
    def set_center(self, center: core.Tensor) -> None:
        """Set the center of the box.
        
        Args:
            center (core.Tensor): Tensor with {3,} shape, and type float32 or float64."""
        ...
    def set_color(self, color: core.Tensor) -> None:
        """Set the color of the oriented box.
        
        Args:
            color (core.Tensor): Tensor with {3,} shape, and type float32 or float64, with values in range [0.0, 1.0]."""
        ...
    def set_extent(self, extent: core.Tensor) -> None:
        """Set the extent of the box.
        
        Args:
            extent (core.Tensor): Tensor with {3,} shape, and type float32 or float64."""
        ...
    def set_rotation(self, rotation: core.Tensor) -> None:
        """Set the rotation matrix of the box.
        
        Args:
            rotation (core.Tensor): Tensor with {3, 3} shape, and type float32 or float64."""
        ...
    def to(self, device: core.Device, copy: bool = False) -> OrientedBoundingBox:
        """Transfer the oriented box to a specified device.
        
        Args:
            device (core.Device):
            copy (bool, optional): Defaults to False.
        
        Returns:
            OrientedBoundingBox"""
        ...
    def to_legacy(self) -> geometry.OrientedBoundingBox: ...
    def transform(self, transformation: core.Tensor) -> OrientedBoundingBox:
        """Transform the oriented box by the given transformation matrix.
        
        Args:
            transformation (core.Tensor): Transformation matrix of shape {4, 4}, type float32 or float64, device same as the box.
        
        Returns:
            OrientedBoundingBox"""
        ...
    def translate(self, translation: core.Tensor, relative: bool = True) -> OrientedBoundingBox:
        """Translate the oriented box by the given translation. If relative is true, the translation is added to the center of the box. If false, the center will be assigned to the translation.
        
        Args:
            translation (core.Tensor): Translation tensor of shape {3,}, type float32 or float64, device same as the box.
            relative (bool, optional): Whether to perform relative translation. Defaults to True.
        
        Returns:
            OrientedBoundingBox"""
        ...

class AxisAlignedBoundingBox:
    @property
    def color(self) -> core.Tensor:
        """Returns the color for box."""
        ...
    @property
    def device(self) -> core.Device:
        """Returns the device of the geometry."""
        ...
    @property
    def dtype(self) -> core.Dtype:
        """Returns the data type attribute of this AxisAlignedBoundingBox."""
        ...
    @property
    def is_cpu(self) -> bool:
        """Returns true if the geometry is on CPU."""
        ...
    @property
    def is_cuda(self) -> bool:
        """Returns true if the geometry is on CUDA."""
        ...
    @property
    def material(self) -> Any: # todo
        ...
    @property
    def max_bound(self) -> core.Tensor:
        """Returns the max bound for box coordinates."""
        ...
    @property
    def min_bound(self) -> core.Tensor:
        """Returns the min bound for box coordinates."""
        ...

    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> AxisAlignedBoundingBox:
        """Clear all elements in the geometry."""
        ...
    def clone(self) -> AxisAlignedBoundingBox:
        """Returns copy of the axis-aligned box on the same device."""
        ...
    def cpu(self) -> AxisAlignedBoundingBox:
        """Transfer the axis-aligned box to CPU. If the axis-aligned box is already on CPU, no copy will be performed."""
        ...
    def create_from_points(points: core.Tensor) -> AxisAlignedBoundingBox:
        """Creates the axis-aligned box that encloses the set of points.
        
        Args:
            points (core.Tensor): A list of points with data type of float32 or float64 (N x 3 tensor).
            
        Returns:
            AxisAlignedBoundingBox"""
        ...
    def cuda(self, device_id: int = 0) -> AxisAlignedBoundingBox:
        """Transfer the axis-aligned box to a CUDA device. If the axis-aligned box is already on the specified CUDA device, no copy will be performed.
        
        Args:
            device_id (int, optional): Defaults to 0.
        
        Returns:
            AxisAlignedBoundingBox"""
        ...
    @staticmethod
    def from_legacy(
        box: geometry.AxisAlignedBoundingBox,
        dtype: core.Dtype = core.Dtype.Float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> AxisAlignedBoundingBox:
        """Create an AxisAlignedBoundingBox from a legacy Open3D axis-aligned box.
        
        Args:
            box (geometry.AxisAlignedBoundingBox):
            dtype (core.Dtype, optional): Defaults to core.Dtype.Float32.
            device (core.Device, optional): Defaults to core.Device("CPU:0").
            
        Returns:
            AxisAlignedBoundingBox"""
        ...
    def get_box_points(self) -> core.Tensor:
        """Returns the eight points that define the bounding box. The Return tensor has shape {8, 3} and data type of float32."""
        ...
    def get_center(self) -> core.Tensor:
        """Returns the center for box coordinates."""
        ...
    def get_extent(self) -> core.Tensor:
        """Get the extent/length of the bounding box in x, y, and z dimension."""
        ...
    def get_half_extent(self) -> core.Tensor:
        """Returns the half extent of the bounding box."""
        ...
    def get_max_extent(self) -> float:
        """Returns the maximum extent, i.e. the maximum of X, Y and Z axis's extents."""
        ...
    def get_oriented_bounding_box(self) -> OrientedBoundingBox:
        """Convert to an oriented box."""
        ...
    def get_point_indices_within_bounding_box(points: core.Tensor) -> core.Tensor:
        """Indices to points that are within the bounding box.
        
        Args:
            points (core.Tensor): Tensor with {N, 3} shape, and type float32 or float64.
            
        Returns:
            core.Tensor"""
        ...
    def has_valid_material(self) -> bool:
        """Returns true if the geometry's material is valid."""
        ...
    def is_empty(self) -> bool:
        """Returns True iff the geometry is empty."""
        ...
    def scale(self, scale: float, center: Optional[core.Tensor] = None) -> AxisAlignedBoundingBox:
        """Scale the axis-aligned box. If f$mif$ is the min_bound and f$maf$ is the max_bound of the axis aligned bounding box, and f$sf$ and f$cf$ are the provided scaling factor and center respectively, then the new min_bound and max_bound are given by f$mi = c + s (mi - c)f$ and f$ma = c + s (ma - c)f$. The scaling center will be the box center if it is not specified.
        
        Args:
            scale (float): The scale parameter.
            center (Optional[core.Tensor], optional): Center used for the scaling operation. Tensor with {3,} shape, and type float32 or float64
            
        Returns:
            AxisAlignedBoundingBox"""
        ...
    def set_color(self, color: core.Tensor) -> None:
        """Set the color of the axis-aligned box.
        
        Args:
            color (core.Tensor): Tensor with {3,} shape, and type float32 or float64, with values in range [0.0, 1.0]."""
        ...
    def set_max_bound(self, max_bound: core.Tensor) -> None:
        """Set the upper bound of the axis-aligned box.
        
        Args:
            max_bound (core.Tensor): Tensor with {3,} shape, and type float32 or float64."""
        ...
    def set_min_bound(self, min_bound: core.Tensor) -> None:
        """Set the lower bound of the axis-aligned box.
        
        Args:
            min_bound (core.Tensor): Tensor with {3,} shape, and type float32 or float64."""
        ...
    def to(self, device: core.Device, copy: bool = False) -> AxisAlignedBoundingBox:
        """Transfer the axis-aligned box to a specified device.
        
        Args:
            device (core.Device):
            copy (bool, optional): Defaults to False.
            
        Returns:
            AxisAlignedBoundingBox"""
        ...
    def to_legacy(self) -> geometry.AxisAlignedBoundingBox:
        """Convert to a legacy Open3D axis-aligned box."""
        ...
    def translate(self, translation: core.Tensor, relative: bool = True) -> AxisAlignedBoundingBox:
        """Translate the axis-aligned box by the given translation. If relative is true, the translation is applied to the current min and max bound. If relative is false, the translation is applied to make the box's center at the given translation.
        
        Args:
            translation (core.Tensor): Translation tensor of shape (3,), type float32 or float64, device same as the box.
            relative (bool, optional): Whether to perform relative translation.
            
        Returns:
            AxisAlignedBoundingBox"""
        ...
    def volume(self) -> float:
        """Returns the volume of the bounding box."""
        ...

class DrawableGeometry:
    def __init__(self, *args, **kwargs) -> None: ...
    def has_valid_material(self) -> bool: ...
    @property
    def material(self) -> Any: ...

class Geometry:
    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> Geometry: ...
    def is_empty(self) -> bool: ...

class Image(Geometry):
    channels: int
    columns: int
    device: core.Device
    dtype: core.Dtype
    rows: int
    @overload
    def __init__(
        self,
        rows: int = 0,
        cols: int = 0,
        channels: int = 1,
        dtype: core.Dtype = core.Dtype.Float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> None: ...
    @overload
    def __init__(self, tensor: core.Tensor) -> None: ...
    def as_tensor(self) -> core.Tensor: ...
    def clear(self) -> Image: ...
    def clip_transform(
        self, scale: float, min_value: float, max_value: float, clip_fill: float = 0.0
    ) -> Image: ...
    def clone(self) -> Image: ...
    def colorize_depth(
        self, scale: float, min_value: float, max_value: float
    ) -> Image: ...
    def cpu(self) -> Image: ...
    def create_normal_map(self, invalid_fill: float = 0.0) -> Image: ...
    def create_vertex_map(
        self, intrinsics: core.Tensor, invalid_fill: float = 0.0
    ) -> Image: ...
    def cuda(self, device_id: int = 0) -> Image: ...
    def dilate(self, kernel_size: int = 3) -> Image: ...
    def filter(self, kernel: core.Tensor) -> Image: ...
    def filter_bilateral(
        self, kernel_size: int = 3, value_sigma: float = 20.0, dist_sigma: float = 10.0
    ) -> Image: ...
    def filter_gaussian(self, kernel_size: int = 3, sigma: float = 1.0) -> Image: ...
    def filter_sobel(self, kernel_size: int = 3) -> tuple[Image, Image]: ...
    @classmethod
    def from_legacy_image(
        cls, image_legacy: geometry.Image, device: core.Device = core.Device("CPU:0")
    ) -> Image: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def linear_transform(self, scale: float = 1.0, offset: float = 0.0) -> Image: ...
    def pyrdown(self) -> Image: ...
    def resize(
        self, sampling_rate: float = 0.5, interp_type: InterpType = InterpType.Nearest
    ) -> Image: ...
    def rgb_to_gray(self) -> Image: ...
    @overload
    def to(self, device: core.Device, copy: bool = False) -> Image: ...
    @overload
    def to(
        self,
        dtype: core.Dtype,
        scale: Optional[float] = None,
        offset: float = 0.0,
        copy: bool = False,
    ) -> Image: ...
    def to_legacy_image(self) -> geometry.Image: ...

class InterpType(Enum):
    Cubic = ...
    Lanczos = ...
    Linear = ...
    Nearest = ...
    Super = ...

class PointCloud(Geometry):
    point: TensorMap
    @overload
    def __init__(self, device: core.Device) -> None: ...
    @overload
    def __init__(self, points: core.Tensor) -> None: ...
    @overload
    def __init__(self, map_keys_to_tensors: dict[str, core.Tensor]) -> None: ...
    def append(self, other: PointCloud) -> PointCloud: ...
    def clone(self) -> PointCloud: ...
    def cpu(self) -> PointCloud: ...
    @classmethod
    def create_from_depth_image(
        cls,
        depth: Image,
        intrinsics: core.Tensor,
        extrinsics: core.Tensor = ...,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        stride: int = 1,
        with_normals: bool = False,
    ) -> PointCloud: ...
    @classmethod
    def create_from_rgbd_image(
        cls,
        rgbd_image: RGBDImage,
        intrinsics: core.Tensor,
        extrinsics: core.Tensor = ...,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        stride: int = 1,
        with_normals: bool = False,
    ) -> PointCloud: ...
    def cuda(self, device_id: int = 0) -> PointCloud: ...
    @classmethod
    def from_legacy(
        cls,
        pcd_legacy: geometry.PointCloud,
        dtype: core.Dtype = core.Dtype.Float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> PointCloud: ...
    def get_center(self) -> core.Tensor: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def rotate(self, R: core.Tensor, center: core.Tensor) -> PointCloud: ...
    def scale(self, scale: float, center: core.Tensor) -> PointCloud: ...
    def to(self, device: core.Device, copy: bool = False) -> PointCloud: ...
    def to_legacy(self) -> geometry.PointCloud: ...
    def transform(self, transformation: core.Tensor) -> PointCloud: ...
    def translate(
        self, translation: core.Tensor, relative: bool = True
    ) -> PointCloud: ...
    def voxel_down_sample(self, voxel_size: float) -> PointCloud: ...

class RGBDImage(Geometry):
    aligned_: bool
    color: Image
    depth: Image
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, color: Image, depth: Image, aligned: bool = True) -> None: ...
    def are_aligned(self) -> bool: ...
    def clear(self) -> RGBDImage: ...
    def clone(self) -> RGBDImage: ...
    def cpu(self) -> RGBDImage: ...
    def cuda(self, device_id: int = 0) -> RGBDImage: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def to(self, device: core.Device, copy: bool = False) -> RGBDImage: ...
    def to_legacy(self) -> geometry.RGBDImage: ...

class RaycastingScene:
    INVALID_ID: int = 4294967295

    def __init__(self, nthreads: int = 0) -> None:
        """Create a RaycastingScene.
        
        Args:
            nthreads (int, optional): The number of threads to use for building the scene. Set to 0 for automatic."""
        ...
    @overload
    def add_triangles(self, vertex_positions: core.Tensor, triangle_indices: core.Tensor) -> int:
        """Add a triangle mesh to the scene.
        
        Args:
            vertices (core.Tensor): Vertices as Tensor of dim {N,3} and dtype Float32.
            triangles (core.Tensor): Triangles as Tensor of dim {M,3} and dtype UInt32.
        
        Returns:
            int: The geometry ID of the added mesh."""
        ...
    @overload
    def add_triangles(self, mesh: TriangleMesh) -> int:
        """Add a triangle mesh to the scene.
        
        Args:
            mesh (o3d.t.geometry.TriangleMesh): A triangle mesh.
        
        Returns:
            int: The geometry ID of the added mesh."""
        ...
    def cast_rays(self, rays: core.Tensor, nthreads: int = 0) -> dict[str, core.Tensor]:
        """Computes the first intersection of the rays with the scene.
        
        Args:
            rays (core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays. {..} can be any number of dimensions, e.g., to organize rays for creating an image the shape can be {height, width, 6}. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction but the returned hit distance uses the length of the direction vector as unit.
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
        
        Returns:
            dict[str, core.Tensor]: A dictionary which contains the following keys
                t_hit: A tensor with the distance to the first hit. The shape is {..}. If there is no intersection the hit distance is inf.
                geometry_ids: A tensor with the geometry IDs. The shape is {..}. If there is no intersection the ID is INVALID_ID.
                primitive_ids: A tensor with the primitive IDs, which corresponds to the triangle index. The shape is {..}. If there is no intersection the ID is INVALID_ID.
                primitive_uvs: A tensor with the barycentric coordinates of the hit points within the hit triangles. The shape is {.., 2}.
                primitive_normals: A tensor with the normals of the hit triangles. The shape is {.., 3}."""
        ...
    def compute_closest_points(self, query_points: core.Tensor, nthreads: int = 0) -> dict[str, core.Tensor]: 
        """Computes the closest points on the surfaces of the scene.
        
        Args:
            query_points (core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query_point to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
        
        Returns:
            dict[str, core.Tensor]: The returned dictionary contains
                points: A tensor with the closest surface points. The shape is {..}.
                geometry_ids: A tensor with the geometry IDs. The shape is {..}.
                primitive_ids: A tensor with the primitive IDs, which corresponds to the triangle index. The shape is {..}.
                primitive_uvs: A tensor with the barycentric coordinates of the closest points within the triangles. The shape is {.., 2}.
                primitive_normals: A tensor with the normals of the closest triangle . The shape is {.., 3}."""
        ...
    def compute_distance(self, query_points: core.Tensor, nthreads: int = 0) -> core.Tensor:
        """Computes the distance to the surface of the scene.
        
        Args:
            query_points (core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query points to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
        
        Returns:
            core.Tensor: A tensor with the distances to the surface. The shape is {..}."""
        ...
    def compute_occupancy(self, query_points: core.Tensor, nthreads: int = 0, nsamples: int = 1) -> core.Tensor:
        """Computes the occupancy at the query point positions.
        
        This function computes whether the query points are inside or outside. The function assumes that all meshes are watertight and that there are no intersections between meshes, i.e., inside and outside must be well defined. The function determines if a point is inside by counting the intersections of a rays starting at the query points.
        
        Args:
            query_points (core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query points to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
            nsamples (int, optional): The number of rays used for determining the inside. This must be an odd number. The default is 1. Use a higher value if you notice errors in the occupancy values. Errors can occur when rays hit exactly an edge or vertex in the scene.
        
        Returns:
            core.Tensor: A tensor with the occupancy values. The shape is {..}. Values are either 0 or 1. A point is occupied or inside if the value is 1."""
        ...
    def compute_signed_distance(self, query_points: core.Tensor, nthreads: int = 0, nsamples: int = 1) -> core.Tensor:
        """Computes the signed distance to the surface of the scene.
        
        This function computes the signed distance to the meshes in the scene. The function assumes that all meshes are watertight and that there are no intersections between meshes, i.e., inside and outside must be well defined. The function determines the sign of the distance by counting the intersections of a rays starting at the query points.
        
        Args:
            query_points (core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query_points. {..} can be any number of dimensions, e.g., to organize the query points to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
            nsamples (int, optional): The number of rays used for determining the inside. This must be an odd number. The default is 1. Use a higher value if you notice sign flipping, which can occur when rays hit exactly an edge or vertex in the scene.
        
        Returns:
            core.Tensor: A tensor with the signed distances to the surface. The shape is {..}. Negative distances mean a point is inside a closed surface."""
        ...
    def count_intersections(self, rays: core.Tensor, nthreads: int = 0) -> core.Tensor:
        """Computes the number of intersection of the rays with the scene.
        
        Args:
            rays (core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays. {..} can be any number of dimensions, e.g., to organize rays for creating an image the shape can be {height, width, 6}. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction.
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
        
        Returns:
            core.Tensor: A tensor with the number of intersections. The shape is {..}."""
        ...

    @overload
    @staticmethod
    def create_rays_pinhole(
        intrinsic_matrix: core.Tensor,
        extrinsic_matrix: core.Tensor,
        width_px: int,
        height_px: int,
    ) -> core.Tensor:
        """Creates rays for the given camera parameters.
        
        Args:
            intrinsic_matrix (core.Tensor): The upper triangular intrinsic matrix with shape {3,3}.
            extrinsic_matrix (core.Tensor): The 4x4 world to camera SE(3) transformation matrix.
            width_px (int): The width of the image in pixels.
            height_px (int): The height of the image in pixels.
        
        Returns:
            core.Tensor: A tensor of shape {height_px, width_px, 6} with the rays."""
        ...
    @overload
    @staticmethod
    def create_rays_pinhole(
        fov_deg: float,
        center: core.Tensor,
        eye: core.Tensor,
        up: core.Tensor,
        width_px: int,
        height_px: int,
    ) -> core.Tensor:
        """Creates rays for the given camera parameters.
        
        Args:
            fov_deg (float): The horizontal field of view in degree.
            center (core.Tensor): The point the camera is looking at with shape {3}.
            eye (core.Tensor): The position of the camera with shape {3}.
            up (core.Tensor): The up-vector with shape {3}.
            width_px (int): The width of the image in pixels.
            height_px (int): The height of the image in pixels.
        
        Returns:
            core.Tensor: A tensor of shape {height_px, width_px, 6} with the rays."""
        ...
    def list_intersections(self, rays: core.Tensor, nthreads: int = 0) -> dict[str, core.Tensor]:
        """Lists the intersections of the rays with the scene.
        
        Args:
            rays (core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays; {..} can be any number of dimensions. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction although it should be normalised if t_hit is to be calculated in coordinate units.
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
        
        Returns:
            dict[str, core.Tensor]: The returned dictionary contains
                ray_splits: A tensor with ray intersection splits. Can be used to iterate over all intersections for each ray. The shape is {num_rays + 1}.
                ray_ids: A tensor with ray IDs. The shape is {num_intersections}.
                t_hit: A tensor with the distance to the hit. The shape is {num_intersections}.
                geometry_ids: A tensor with the geometry IDs. The shape is {num_intersections}.
                primitive_ids: A tensor with the primitive IDs, which corresponds to the triangle index. The shape is {num_intersections}.
                primitive_uvs: A tensor with the barycentric coordinates of the intersection points within the triangles. The shape is {num_intersections, 2}."""
        ...
    def test_occlusions(self, rays: core.Tensor, tnear: float = 0.0, tfar: float = inf, nthreads: int = 0) -> core.Tensor:
        """Checks if the rays have any intersection with the scene.
        
        Args:
            rays (core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays. {..} can be any number of dimensions, e.g., to organize rays for creating an image the shape can be {height, width, 6}. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction.
            tnear (float, optional): The tnear offset for the rays. The default is 0.
            tfar (float, optional): The tfar value for the ray. The default is infinity.
            nthreads (int, optional): The number of threads to use. Set to 0 for automatic.
        
        Returns:
            core.Tensor: A boolean tensor which indicates if the ray is occluded by the scene (true) or not (false)."""
        ...

class SurfaceMaskCode(Enum):
    ColorMap = ...
    DepthMap = ...
    NormalMap = ...
    VertexMap = ...

class TensorMap:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, primary_key: str) -> None: ...
    @overload
    def __init__(
        self, primary_key: str, map_keys_to_tensors: dict[str, core.Tensor]
    ) -> None: ...
    def assert_size_synchronized(self) -> None: ...
    def erase(self, key: str) -> int: ...
    def get_primary_key(self) -> str: ...
    def is_size_synchronized(self) -> bool: ...
    def items(self) -> Iterator: ...
    def __getitem__(self, key: str) -> core.Tensor: ...
    def __setitem__(self, key: str, value: core.Tensor) -> TensorMap: ...

class TriangleMesh(Geometry):
    triangle: TensorMap
    vertex: TensorMap
    @overload
    def __init__(self, device: core.Device = core.Device("CPU:0")) -> None: ...
    @overload
    def __init__(
        self, vertex_positions: core.Tensor, triangle_indices: core.Tensor
    ) -> None: ...
    def clear(self) -> TriangleMesh: ...
    def clone(self) -> TriangleMesh: ...
    def cpu(self) -> TriangleMesh: ...
    def cuda(self, device_id: int = 0) -> TriangleMesh: ...
    @classmethod
    def from_legacy(
        cls,
        mesh_legacy: geometry.TriangleMesh,
        vertex_dtype: core.Dtype = core.float32,
        triangle_dtype: core.Dtype = core.int64,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    def get_center(self) -> core.Tensor: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def has_valid_material(self) -> bool: ...
    def rotate(self, R: core.Tensor, center: core.Tensor) -> TriangleMesh: ...
    def scale(self, scale: float, center: core.Tensor) -> TriangleMesh: ...
    def to(self, device: core.Device, copy: bool = False) -> TriangleMesh: ...
    def to_legacy(self) -> geometry.TriangleMesh: ...
    def transform(self, transformation: core.Tensor) -> TriangleMesh: ...
    def translate(
        self, translation: core.Tensor, relative: bool = True
    ) -> TriangleMesh: ...

class VoxelBlockGrid:
    def __init__(
        self,
        attr_names: Sequence[str],
        attr_dtypes: Sequence[core.Dtype],
        attr_channels: Sequence[Union[Iterable, int]],
        voxel_size: float = 0.0058,
        block_resolution: int = 16,
        block_count: int = 10000,
        device: core.Device = core.Device("CPU:0"),
    ) -> None: ...
    def attribute(self, attribute_name: str) -> core.Tensor: ...
    @overload
    def compute_unique_block_coordinates(
        self,
        depth: Image,
        intrisic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> core.Tensor: ...
    @overload
    def compute_unique_block_coordinates(
        self,
        pcd: PointCloud,
        trunc_voxel_multiplier: float = 8.0,
    ) -> core.Tensor: ...
    def extract_point_cloud(
        self, weight_threshold: float = 3.0, estimated_point_number: int = -1
    ) -> PointCloud: ...
    def extract_triangle_mesh(
        self, weight_threshold: float = 3.0, estimated_point_number: int = -1
    ) -> TriangleMesh: ...
    def hashmap(self) -> core.HashMap: ...
    @overload
    def integrate(
        self,
        block_coords: core.Tensor,
        depth: Image,
        color: Image,
        depth_intrinsic: core.Tensor,
        color_intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> None: ...
    @overload
    def integrate(
        self,
        block_coords: core.Tensor,
        depth: Image,
        color: Image,
        intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> None: ...
    @overload
    def integrate(
        self,
        block_coords: core.Tensor,
        depth: Image,
        intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> None: ...
    @classmethod
    def load(cls, file_name: str) -> VoxelBlockGrid: ...
    def ray_cast(
        self,
        block_coords: core.Tensor,
        intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        width: int,
        height: int,
        render_attributes: Sequence[str] = ["depth", "color"],
        depth_scale: float = 1000.0,
        depth_min: float = 0.1,
        depth_max: float = 3.0,
        weight_threshold: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
        range_map_down_factor: int = 8,
    ) -> TensorMap: ...
    def save(self, file_name: str) -> None: ...
    def voxel_coordinates(self, voxel_indices: core.Tensor) -> core.Tensor: ...
    @overload
    def voxel_coordinates_and_flattened_indices(
        self, buf_indices: core.Tensor
    ) -> tuple[core.Tensor, core.Tensor]: ...
    @overload
    def voxel_coordinates_and_flattened_indices(
        self,
    ) -> tuple[core.Tensor, core.Tensor]: ...
    @overload
    def voxel_indices(self, buf_indices: core.Tensor) -> core.Tensor: ...
    @overload
    def voxel_indices(self) -> core.Tensor: ...
