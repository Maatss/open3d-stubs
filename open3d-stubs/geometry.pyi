from enum import Enum
import sys
from typing import Callable, Tuple, overload

from numpy import array, float64, int32
from numpy.typing import ArrayLike, NDArray

from . import camera, pipelines, utility

# pylint: skip-file

class Geometry:
    class GeometryType(Enum):
        Unspecified = 0
        PointCloud = 1
        VoxelGrid = 2
        Octree = 3
        LineSet = 4
        MeshBase = 5
        TriangleMesh = 6
        HalfEdgeTriangleMesh = 7
        Image = 8
        RGBDImage = 9
        TetraMesh = 10
        OrientedBoundingBox = 11
        AxisAlignedBoundingBox = 12
    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> Geometry: ...
    def dimension(self) -> int: ...
    def get_geometry_type(self) -> Geometry.GeometryType: ...
    def is_empty(self) -> bool: ...

class Geometry2D(Geometry):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_max_bound(self) -> NDArray[float64]: ...
    def get_min_bound(self) -> NDArray[float64]: ...

class Geometry3D(Geometry):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_axis_aligned_bounding_box(self) -> AxisAlignedBoundingBox: ...
    def get_center(self) -> NDArray[float64]: ...
    def get_max_bound(self) -> NDArray[float64]: ...
    def get_min_bound(self) -> NDArray[float64]: ...
    def get_oriented_bounding_box(self) -> OrientedBoundingBox: ...
    @classmethod
    def get_rotation_matrix_from_axis_angle(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_quaternion(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_xyz(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_xzy(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_yxz(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_yzx(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_zxy(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    @classmethod
    def get_rotation_matrix_from_zyx(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...
    def rotate(
        self, R: NDArray[float64], center: NDArray[float64] = ...
    ) -> Geometry3D: ...
    def scale(self, scale: float, center: NDArray[float64]) -> Geometry3D: ...
    def transform(self, transformation: NDArray[float64]) -> Geometry3D: ...
    def translate(
        self, translation: NDArray[float64], relative: bool = True
    ) -> Geometry3D: ...

class PointCloud(Geometry3D):
    colors: utility.Vector3dVector
    normals: utility.Vector3dVector
    points: utility.Vector3dVector
    def __init__(self, *args, **kwargs): ...
    def __add__(self, cloud: PointCloud) -> PointCloud: ...
    def __iadd__(self, cloud: PointCloud) -> PointCloud: ...
    def cluster_dbscan(
        self, eps: float, min_points: int, print_progress: bool = False
    ) -> utility.IntVector: ...
    def compute_convex_hull(self) -> tuple[TriangleMesh, list[int]]: ...
    def compute_mahalanobis_distance(self) -> utility.DoubleVector: ...
    def compute_mean_and_covariance(
        self,
    ) -> tuple[NDArray[float64], NDArray[float64]]: ...
    def compute_nearest_neighbor_distance(self) -> utility.DoubleVector: ...
    def compute_point_cloud_distance(
        self, target: PointCloud
    ) -> utility.DoubleVector: ...
    @classmethod
    def create_from_depth_image(
        cls,
        depth: Image,
        intrinsic,
        extrinsic: NDArray[float64],
        depth_scale: float = 1000.0,
        depth_trunc: float = 1000.0,
        stride: int = 1,
        project_valid_depth_only: bool = True,
    ) -> PointCloud: ...
    @classmethod
    def create_from_rgbd_image(
        cls,
        iamge: Image,
        intrinsic,
        extrinsic: NDArray[float64],
        project_valid_depth_only: bool = True,
    ) -> PointCloud: ...
    def crop(
        self,
        bounding_box: AxisAlignedBoundingBox | OrientedBoundingBox,
    ) -> PointCloud: ...
    def estimate_normals(
        self,
        search_param: KDTreeSearchParam = KDTreeSearchParamKNN(),
        fast_normal_computation: bool = True,
    ) -> None: ...
    def has_colors(self) -> bool: ...
    def has_normals(self) -> bool: ...
    def has_points(self) -> bool: ...
    def hidden_point_removal(
        self, camera_location: ArrayLike, radius: float
    ) -> tuple[TriangleMesh, list[int]]: ...
    def normalize_normals(self) -> PointCloud: ...
    def orient_normals_consistent_tangent_plane(self, k: int) -> None: ...
    def orient_normals_to_align_with_direction(
        self, orientation_reference: NDArray[float64] = array([0.0, 0.0, 1.0])
    ) -> None: ...
    def orient_normals_towards_camera_location(
        self, camera_location: NDArray[float64] = array([0.0, 0.0, 0.0])
    ) -> None: ...
    def paint_uniform_color(self, color: ArrayLike) -> PointCloud: ...
    def random_down_sample(self, sampling_ratio: float) -> PointCloud: ...
    def remove_non_finite_points(
        self, remove_nan: bool = True, remove_infinite: bool = True
    ) -> PointCloud: ...
    def remove_radius_outlier(
        self, nb_potins: int, radius: float
    ) -> tuple[PointCloud, list[int]]: ...
    def remove_statistical_outlier(
        self, nb_neighbors: int, std_ratio: float
    ) -> tuple[PointCloud, list[int]]: ...
    def segment_plane(
        self, distance_threshold: float, ransac_n: int, num_iterations: int
    ) -> tuple[NDArray[float64], list[int]]: ...
    def select_by_index(
        self, indices: list[int], invert: bool = False
    ) -> PointCloud: ...
    def uniform_down_sample(self, every_k_points: int) -> PointCloud: ...
    def voxel_down_sample(self, voxel_size: float) -> PointCloud: ...
    def voxel_down_sample_and_trace(
        self,
        voxel_size: float,
        min_bound: NDArray[float64],
        max_bound: NDArray[float64],
        approximate_class: bool = False,
    ) -> tuple[PointCloud, NDArray[int32], list[utility.IntVector]]: ...
    def rotate(
        self, R: NDArray[float64], center: NDArray[float64] = ...
    ) -> PointCloud: ...
    def scale(self, scale: float, center: NDArray[float64]) -> PointCloud: ...
    def transform(self, transformation: NDArray[float64]) -> PointCloud: ...
    def translate(
        self, translation: NDArray[float64], relative: bool = True
    ) -> PointCloud: ...

class Image(Geometry2D):
    def __init__(self, *args, **kwargs) -> None: ...
    def create_pyramid(
        self, num_of_levels: int, with_gaussian_filter: bool
    ) -> list[Image]: ...
    def filter(self, filter_type: ImageFilterType) -> Image: ...
    @classmethod
    def filter_pyramid(
        cls, image_pyramid: list[Image], filter_type: ImageFilterType
    ) -> list[Image]: ...
    def flip_horizontal(self) -> Image: ...
    def flip_vertical(self) -> Image: ...

class ImageFilterType(Enum):
    Gaussian5: ...
    Gaussian7: ...
    Sobel3dx: ...
    Sobel3dy: ...

class KDTreeFlann:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, data: NDArray[float64]) -> None: ...
    @overload
    def __init__(self, geometry: Geometry) -> None: ...
    @overload
    def __init__(self, feature: pipelines.registration.Feature) -> None: ...
    def search_hybrid_vector_3d(
        self, query: ArrayLike, radius: float, max_nn: int
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_hybrid_vector_xd(
        self, query: ArrayLike, radius: float, max_nn: int
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_knn_vector_3d(
        self, query: ArrayLike, max_nn: int
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_knn_vector_xd(
        self, query: ArrayLike, max_nn: int
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_radius_vector_3d(
        self, query: ArrayLike, radius: float
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_radius_vector_xd(
        self, query: ArrayLike, radius: float
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_vector_3d(
        self, query: ArrayLike, search_param: KDTreeSearchParam
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def search_vector_xd(
        self, query: ArrayLike, search_param: KDTreeSearchParam
    ) -> tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def set_feature(self, feature: pipelines.registration.Feature) -> bool: ...
    def set_geometry(self, geometry: Geometry) -> bool: ...
    def set_matrix_data(self, data: NDArray[float64]) -> bool: ...

class KDTreeSearchParam:
    class SearchType(Enum):
        HybridSearch: ...
        KNNSearch: ...
        RadiusSearch: ...
    def __init__(self, *args, **kwargs) -> None: ...
    def get_search_type(self) -> KDTreeSearchParam.SearchType: ...

class KDTreeSearchParamHybrid(KDTreeSearchParam):
    max_nn: int
    radius: float
    def __init__(self, radius: float, max_nn: int) -> None: ...

class KDTreeSearchParamKNN(KDTreeSearchParam):
    knn: int
    def __init__(self, knn: int = 30) -> None: ...

class KDTreeSearchParamRadius(KDTreeSearchParam):
    radius: float
    def __init__(self, radius: float) -> None: ...

class AxisAlignedBoundingBox(Geometry3D):
    color: NDArray[float64]
    max_bound: NDArray[float64]
    min_bound: NDArray[float64]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, aabb: AxisAlignedBoundingBox) -> None: ...
    @overload
    def __init__(
        self, min_bound: NDArray[float64], max_bound: NDArray[float64]
    ) -> None: ...
    @classmethod
    def create_from_points(
        cls, points: utility.Vector3dVector
    ) -> AxisAlignedBoundingBox: ...
    def get_box_points(self) -> utility.Vector3dVector: ...
    def get_extent(self) -> NDArray[float64]: ...
    def get_half_extent(self) -> NDArray[float64]: ...
    def get_max_extent(self) -> float: ...
    def get_point_indices_within_bounding_box(
        self, points: utility.Vector3dVector
    ) -> list[int]: ...
    def get_print_info(self) -> str: ...
    def volume(self) -> float: ...

class OrientedBoundingBox(Geometry3D):
    color: ArrayLike
    max_bound: ArrayLike
    min_bound: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def create_from_axis_aligned_bounding_box(
        cls, aabox: AxisAlignedBoundingBox
    ) -> OrientedBoundingBox: ...
    @classmethod
    def create_from_points(
        cls, points: utility.Vector3dVector
    ) -> OrientedBoundingBox: ...
    def get_box_points(self) -> utility.Vector3dVector: ...
    def get_point_indices_within_bounding_box(
        self, points: utility.Vector3dVector
    ) -> list[int]: ...
    def volume(self) -> float: ...

class LineSet(Geometry3D):
    colors: ArrayLike
    lines: ArrayLike
    points: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    @classmethod
    def create_camera_visualization(
        cls,
        view_width_px: int,
        view_height_px: int,
        intrinsic: NDArray[float64],
        extrinsic: NDArray[float64],
        scale: float = 1.0,
    ) -> LineSet: ...
    @overload
    @classmethod
    def create_camera_visualization(
        cls,
        intrinsic: camera.PinholeCameraIntrinsic,
        extrinsic: NDArray[float64],
        scale: float = 1.0,
    ) -> LineSet: ...
    @classmethod
    def create_from_axis_aligned_bounding_box(
        cls, box: AxisAlignedBoundingBox
    ) -> LineSet: ...
    @classmethod
    def create_from_oriented_bounding_box(cls, box: OrientedBoundingBox) -> LineSet: ...
    @classmethod
    def create_from_point_cloud_correspondences(
        cls,
        cloud0: PointCloud,
        cloud1: PointCloud,
        correspondences: list[tuple[int, int]],
    ) -> LineSet: ...
    @classmethod
    def create_from_tetra_mesh(cls, mesh: TetraMesh) -> LineSet: ...
    @classmethod
    def create_from_triangle_mesh(cls, mesh: TriangleMesh) -> LineSet: ...
    def paint_uniform_color(self, color: ArrayLike) -> LineSet: ...
    def has_colors(self) -> bool: ...
    def has_lines(self) -> bool: ...
    def has_points(self) -> bool: ...

class MeshBase(Geometry3D):
    vertex_colors: utility.Vector3dVector
    vertex_normals: utility.Vector3dVector
    vertices: utility.Vector3dVector
    def __init__(self, *args, **kwargs) -> None: ...
    def compute_convex_hull(self) -> tuple[TriangleMesh, list[int]]: ...
    def has_vertex_colors(self) -> bool: ...
    def has_vertex_normals(self) -> bool: ...
    def has_vertices(self) -> bool: ...
    def normalize_normals(self) -> MeshBase: ...

class TriangleMesh(MeshBase):
    adjacency_list: list[set]
    textures: Image
    triangle_material_ids: utility.IntVector
    triangle_normals: utility.Vector3dVector
    triangle_uvs: utility.Vector2dVector
    triangles: utility.Vector3iVector

    @overload
    def __init__(self) -> None:
        """Default constructor."""
        ...

    @overload
    def __init__(self, other: TriangleMesh) -> None:
        """Copy constructor."""
        ...
    
    @overload
    def __init__(
        self,
        vertices: utility.Vector3dVector,
        triangles: utility.Vector3iVector,
    ) -> None:
        """Create a triangle mesh from vertices and triangle indices."""
        ...

    def cluster_connected_triangles(
        self,
    ) -> tuple[utility.IntVector, list[int], utility.DoubleVector]: ...
    def compute_adjacency_list(self) -> TriangleMesh: ...
    def compute_triangle_normals(self, normalized: bool = True) -> TriangleMesh: ...
    def compute_vertex_normals(self, normalized: bool = True) -> TriangleMesh: ...
    @classmethod
    def create_arrow(
        cls,
        cylinder_radius: float = 1.0,
        cone_radius: float = 1.5,
        cylinder_height: float = 5.0,
        cone_height: float = 4.0,
        resolution: int = 20,
        cylinder_split: int = 4,
        cone_split: int = 1,
    ) -> TriangleMesh: ...
    @classmethod
    def create_box(
        cls,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        create_uv_map: bool = False,
        map_texturea_to_each_face: bool = False,
    ) -> TriangleMesh: ...
    @classmethod
    def create_cone(
        cls,
        radius: float = 1.0,
        height: float = 2.0,
        resolution: int = 20,
        split: int = 1,
        create_uv_map: bool = False,
    ) -> TriangleMesh: ...
    @classmethod
    def create_coordinate_frame(
        cls, size: float = 1.0, origin: ArrayLike = array([0.0, 0.0, 0.0])
    ) -> TriangleMesh: ...
    @classmethod
    def create_cylinder(
        cls,
        radius: float = 1.0,
        height: float = 2.0,
        resolution: int = 20,
        split: int = 4,
        create_uv_map: bool = False,
    ) -> TriangleMesh: ...

    @classmethod
    def create_from_oriented_bounding_box(
        cls,
        obox: OrientedBoundingBox,
        scale: NDArray[float64] = array([1.0, 1.0, 1.0]),
        create_uv_map: bool = False,
    ) -> TriangleMesh:
        """Factory function to create a solid oriented bounding box.
        
        Args:
            obox (OrientedBoundingBox): OrientedBoundingBox object to create mesh of.
            scale (NDArray[float64], optional, default=array([1.0, 1.0, 1.0])):
                scale factor along each direction of OrientedBoundingBox
            create_uv_map (bool, optional, default=False): Add default UV map to the mesh.
        Returns:
            TriangleMesh: Solid oriented bounding box.
        """
        ...
    
    @overload
    @classmethod
    def create_from_point_cloud_alpha_shape(
        cls,
        pcd: PointCloud,
        alpha: float,
    ) -> TriangleMesh:
        """Alpha shapes are a generalization of the convex hull.
        With decreasing alpha value the shape schrinks and creates cavities.
        See Edelsbrunner and Muecke, “Three-Dimensional Alpha Shapes”, 1994.
        
        Args:
            pcd (PointCloud): PointCloud from which the TriangleMesh surface is reconstructed.
            alpha (float): Parameter to control the shape. A very big value will give a shape
                close to the convex hull.
        Returns:
            TriangleMesh: Surface of the alpha shape.
        """
        ...
    
    @overload
    @classmethod
    def create_from_point_cloud_alpha_shape(
        cls,
        pcd: PointCloud,
        alpha: float,
        tetra_mesh: TetraMesh,
        pt_map: list[int] = None,
    ) -> TriangleMesh:
        """Alpha shapes are a generalization of the convex hull.
        With decreasing alpha value the shape shrinks and creates cavities.
        See Edelsbrunner and Muecke, “Three-Dimensional Alpha Shapes”, 1994.
        
        Args:
            pcd (PointCloud): PointCloud from which the TriangleMesh surface is reconstructed.
            alpha (float): Parameter to control the shape. A very big value will give a shape
                close to the convex hull.
            tetra_mesh (TetraMesh): If not None, than uses this to construct the alpha shape.
                Otherwise, TetraMesh is computed from pcd.
            pt_map (list[int], optional, default=None): Optional map from tetra_mesh vertex indices
                to pcd points.
        Returns:
            TriangleMesh: Surface of the alpha shape.
        """

    @classmethod
    def create_from_point_cloud_ball_pivoting(
        cls,
        pcd: PointCloud,
        radii: utility.DoubleVector,
    ) -> TriangleMesh:
        """Function that computes a triangle mesh from a oriented PointCloud.
        This implements the Ball Pivoting algorithm proposed in F. Bernardini et al.,
        “The ball-pivoting algorithm for surface reconstruction”, 1999. The implementation
        is also based on the algorithms outlined in Digne, “An Analysis and Implementation
        of a Parallel Ball Pivoting Algorithm”, 2014. The surface reconstruction is done by
        rolling a ball with a given radius over the point cloud, whenever the ball touches
        three points a triangle is created.
        
        Args:
            pcd (PointCloud): PointCloud from which the TriangleMesh surface is reconstructed. Has to contain normals.
            radii (utility.DoubleVector): The radii of the ball that are used for the surface reconstruction.
        """
        ...
    
    @classmethod
    def create_from_point_cloud_poisson(
        cls,
        pcd: PointCloud,
        depth: int = 8,
        width: int = 0,
        scale: float = 1.1,
        linear_fit: bool = False,
        n_threads: int = -1,
    ) -> Tuple[TriangleMesh, utility.DoubleVector]:
        """Function that computes a triangle mesh from a oriented PointCloud pcd.
        This implements the Screened Poisson Reconstruction proposed in Kazhdan and Hoppe,
        “Screened Poisson Surface Reconstruction”, 2013. This function uses the original
        implementation by Kazhdan. See https://github.com/mkazhdan/PoissonRecon

        Args:
            pcd (PointCloud):  PointCloud from which the TriangleMesh surface is reconstructed. Has to contain normals.
            depth (int, optional, default=8): Maximum depth of the tree that will be used for surface
                reconstruction. Running at depth d corresponds to solving on a grid whose resolution
                is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree
                to the sampling density, the specified reconstruction depth is only an upper bound.
            width (int, optional, default=0): Specifies the target width of the finest level octree cells.
                This parameter is ignored if depth is specified
            scale (float, optional, default=1.1): Specifies the ratio between the diameter of the cube
                used for reconstruction and the diameter of the samples' bounding cube.
            linear_fit (bool, optional, default=False): If true, the reconstructor will use linear
                interpolation to estimate the positions of iso-vertices.
            n_threads (int, optional, default=-1):  Number of threads used for reconstruction.
                Set to -1 to automatically determine it.
        Returns:
            Tuple[TriangleMesh, utility.DoubleVector]: The reconstructed surface and the confidence values
                for each vertex.
        """
        ...

    @classmethod
    def create_icosahedron(
        cls,
        radius: float = 1.0,
        create_uv_map: bool = False,
    ) -> TriangleMesh:
        """Factory function to create a icosahedron.
        The centroid of the mesh will be placed at (0, 0, 0) and the vertices
        have a distance of radius to the center.

        Args:
            radius (float, optional, default=1.0): Distance from centroid to mesh vetices.
            create_uv_map (bool, optional, default=False): Add default uv map to the mesh.
        Returns:
            TriangleMesh: Icosahedron.
        """
        ...

    @classmethod
    def create_mobius(
        cls,
        length_split: int = 70,
        width_split: int = 15,
        twists: int = 1,
        raidus: float = 1,
        flatness: float = 1,
        width: float = 1,
        scale: float = 1,
    ) -> TriangleMesh:
        """Factory function to create a Mobius strip.

        Args:
            length_split (int, optional, default=70): The number of segments along the Mobius strip.
            width_split (int, optional, default=15): The number of segments along the width of the Mobius strip.
            twists (int, optional, default=1): Number of twists of the Mobius strip.
            raidus (float, optional, default=1):
            flatness (float, optional, default=1): Controls the flatness/height of the Mobius strip.
            width (float, optional, default=1): Width of the Mobius strip.
            scale (float, optional, default=1): Scale the complete Mobius strip.
        Returns:
            TriangleMesh: Mobius strip.
        """
        ...

    @classmethod
    def create_octahedron(
        cls,
        radius: float = 1.0,
        create_uv_map: bool = False,
    ) -> TriangleMesh:
        """Factory function to create a octahedron.
        The centroid of the mesh will be placed at (0, 0, 0) and the vertices
        have a distance of radius to the center.

        Args:
            radius (float, optional, default=1.0): Distance from centroid to mesh vertices.
            create_uv_map (bool, optional, default=False): Add default uv map to the mesh.
        Returns:
            TriangleMesh: Octahedron.
        """
        ...

    @classmethod
    def create_sphere(
        cls,
        radius: float = 1.0,
        resolution: int = 20,
        create_uv_map: bool = False,
    ) -> TriangleMesh:
        """Factory function to create a sphere mesh centered at (0, 0, 0).

        Args:
            radius (float, optional, default=1.0): The radius of the sphere.
            resolution (int, optional, default=20): The resolution of the sphere. The longitudes will be
                split into `resolution` segments (i.e. there are `resolution + 1` latitude lines including
                the north and south pole). The latitudes will be split into `2 * resolution` segments
                (i.e. there are `2 * resolution` longitude lines.)
            create_uv_map (bool, optional, default=False): Add default uv map to the mesh.
        Returns:
            TriangleMesh: Sphere.
        """
        ...

    @classmethod
    def create_tetrahedron(
        cls,
        radius: float = 1.0,
        create_uv_map: bool = False,
    ) -> TriangleMesh:
        """Factory function to create a tetrahedron.
        The centroid of the mesh will be placed at (0, 0, 0) and the vertices
        have a distance of radius to the center.

        Args:
            radius (float, optional, default=1.0): Distance from centroid to mesh vertices.
            create_uv_map (bool, optional, default=False): Add default uv map to the mesh.
        Returns:
            TriangleMesh: Tetrahedron.
        """
        ...
    
    @classmethod
    def create_torus(
        cls,
        torus_radius: float = 1.0,
        tube_radius: float = 0.5,
        radial_resolution: int = 30,
        tubular_resolution: int = 20,
    ) -> TriangleMesh:
        """Factory function to create a torus mesh.

        Args:
            torus_radius (float, optional, default=1.0): The radius from the center of the torus to the center of the tube.
            tube_radius (float, optional, default=0.5): The radius of the torus tube.
            radial_resolution (int, optional, default=30): The number of segments along the radial direction.
            tubular_resolution (int, optional, default=20): The number of segments along the tubular direction.
        Returns:
            TriangleMesh: Torus.
        """
        ...

    @overload
    def crop(
        self,
        bounding_box: AxisAlignedBoundingBox,
    ) -> TriangleMesh:
        """Function to crop input TriangleMesh into output TriangleMesh
        
        Args:
            bounding_box (AxisAlignedBoundingBox): AxisAlignedBoundingBox to crop points
        Returns:
            TriangleMesh: Cropped mesh.
        """
        ...

    @overload
    def crop(
        self,
        bounding_box: OrientedBoundingBox,
    ) -> TriangleMesh:
        """Function to crop input TriangleMesh into output TriangleMesh
        
        Args:
            bounding_box (OrientedBoundingBox): OrientedBoundingBox to crop points
        Returns:
            TriangleMesh: Cropped mesh.
        """
        ...

    def deform_as_rigid_as_possible(
        self,
        constraint_vertex_indices: utility.IntVector,
        constraint_vertex_positions: utility.Vector3dVector,
        max_iter: int,
        energy: DeformAsRigidAsPossibleEnergy = DeformAsRigidAsPossibleEnergy.Spokes,
        smoothed_alpha: float = 0.01,
    ) -> TriangleMesh:
        """This function deforms the mesh using the method by Sorkine and Alexa, 'As-Rigid-As-Possible Surface Modeling', 2007
        
        Args:
            constraint_vertex_indices (utility.IntVector): Indices of the triangle vertices
                that should be constrained by the vertex positions in constraint_vertex_positions.
            constraint_vertex_positions (utility.Vector3dVector): Vertex positions used for the constraints.
            max_iter (int): Maximum number of iterations to minimize energy functional.
            energy (DeformAsRigidAsPossibleEnergy, optional, default=DeformAsRigidAsPossibleEnergy.Spokes):
                Energy model that is minimized in the deformation process
            smoothed_alpha (float, optional, default=0.01): trade-off parameter for the smoothed energy
                functional for the regularization term.
        Returns:
            TriangleMesh: Deformed mesh.
        """
        ...

    def euler_poincare_characteristic(self) -> int:
        """Function that computes the Euler-Poincaré characteristic,
        i.e., V + F - E, where V is the number of vertices, F is the number
        of triangles, and E is the number of edges.
        
        Returns:
            int: Euler-Poincare characteristic.
        """
        ...

    def filter_sharpen(
        self,
        number_of_iterations: int = 1,
        strength: float = 1,
        filter_scope: FilterScope = FilterScope.All,
    ) -> TriangleMesh:
        """Function to sharpen triangle mesh. The output value (V_0)
        is the input value (V_i) plus strength times the input value
        minus he sum of he adjacent values. 
        
        Args:
            number_of_iterations (int, optional, default=1): Number of repetitions of this operation
            strength (float, optional, default=1): Filter parameter.
            filter_scope (FilterScope, optional, default=FilterScope.All): 
        Returns:
            TriangleMesh: Sharpened mesh.
        """
        ...

    def filter_smooth_laplacian(
        self,
        number_of_iterations: int = 1,
        lambda_filter: float = 0.5,
        filter_scope: FilterScope = FilterScope.All,
    ) -> TriangleMesh:
        """Function to smooth triangle mesh using Laplacian.

        Args:
            number_of_iterations (int, optional, default=1): Number of repetitions of this operation
            lambda_filter (float, optional, default=0.5): Filter parameter.
            filter_scope (FilterScope, optional, default=FilterScope.All):
        Returns:
            TriangleMesh: Smoothed mesh.
        """
        ...

    def filter_smooth_simple(
        self,
        number_of_iterations: int = 1,
        filter_scope: FilterScope = FilterScope.All,
    ) -> TriangleMesh:
        """Function to smooth triangle mesh with simple neighbour average.

        Args:
            number_of_iterations (int, optional, default=1): Number of repetitions of this operation
            filter_scope (FilterScope, optional, default=FilterScope.All):
        Returns:
            TriangleMesh: Smoothed mesh.
        """
        ...

    def filter_smooth_taubin(
        self,
        number_of_iterations: int = 1,
        lambda_filter: float = 0.5,
        mu: float = -0.53,
        filter_scope: FilterScope = FilterScope.All,
    ) -> TriangleMesh:
        """Function to smooth triangle mesh using method of Taubin, “Curve and Surface
        Smoothing Without Shrinkage”, 1995. Applies in each iteration two times
        filter_smooth_laplacian, first with filter parameter lambda_filter and
        second with filter parameter mu as smoothing parameter.
        This method avoids shrinkage of the triangle mesh.

        Args:
            number_of_iterations (int, optional, default=1): Number of repetitions of this operation
            lambda_filter (float, optional, default=0.5): Filter parameter.
            mu (float, optional, default=-0.53): Filter parameter.
            filter_scope (FilterScope, optional, default=FilterScope.All):
        Returns:
            TriangleMesh: Smoothed mesh.
        """
        ...

    def get_non_manifold_edges(
        self,
        allow_boundary_edges: bool = True
    ) -> utility.Vector2iVector:
        """Get list of non-manifold edges.
        
        Args:
            allow_boundary_edges (bool, optional, default=True): If true, than non-manifold edges are defined
                as edges with more than two adjacent triangles, otherwise each edge that is not adjacent to
                two triangles is defined as non-manifold.
        Returns:
            utility.Vector2iVector: Non-manifold edges.
        """
        ...

    def get_non_manifold_vertices(self) -> utility.IntVector:
        """Returns a list of indices to non-manifold vertices.
        
        Returns:
            utility.IntVector: Non-manifold vertices.
        """
        ...

    def get_self_intersecting_triangles(self) -> utility.Vector2iVector:
        """Returns a list of indices to triangles that intersect the mesh.
        
        Returns:
            utility.Vector2iVector: Self-intersecting triangles.
        """
        ...

    def get_surface_area(self) -> float:
        """Function that computes the surface area of the mesh, i.e. the sum 
        of the individual triangle surfaces.
        
        Returns:
            float: Surface area.
        """
        ...

    def get_volume(self) -> float:
        """Function that computes the volume of the mesh, under the condition
        that it is watertight and orientable.
        
        Returns:
            float: Volume.
        """
        ...

    def has_adjacency_list(self) -> bool: ...
    def has_textures(self) -> bool: ...
    def has_triangle_material_ids(self) -> bool: ...
    def has_triangle_normals(self) -> bool: ...
    def has_triangle_uvs(self) -> bool: ...
    def has_triangles(self) -> bool: ...
    def has_vertex_colors(self) -> bool: ...
    def has_vertex_normals(self) -> bool: ...
    def has_vertices(self) -> bool: ...
    def is_edge_manifold(self, allow_boundary_edges: bool = True) -> bool: ...
    def is_empty(self) -> bool: ...
    def is_intersecting(self, other: TriangleMesh) -> bool: ...
    def is_orientable(self) -> bool: ...
    def is_self_intersecting(self) -> bool: ...
    def is_vertex_manifold(self) -> bool: ...
    def is_watertight(self) -> bool: ...

    def merge_close_vertices(self, eps: float) -> TriangleMesh:
        """Function that will merge close by vertices to a single one. The vertex position,
        normal and color will be the average of the vertices. The parameter eps defines the
        maximum distance of close vertices. This function might help to close triangle soups.
        
        Args:
            eps (float): Parameter that defines the distance between close vertices.
        Returns:
            TriangleMesh: Merged mesh.
        """
        ...

    def orient_triangles(self) -> bool:
        """If the mesh is orientable this function orients all triangles such that all normals
        point towards the same direction.
        
        Returns:
            bool: True if the mesh is orientable.
        """
        ...

    def paint_uniform_color(self, arg0: ArrayLike) -> TriangleMesh: ...

    def remove_degenerate_triangles(self) -> TriangleMesh:
        """Function that removes degenerate triangles, i.e., triangles that references a single
        vertex multiple times in a single triangle. They are usually the product of removing
        duplicated vertices.
        
        Returns:
            TriangleMesh: Mesh without degenerate triangles.
        """
        ...
    
    def remove_duplicated_triangles(self) -> TriangleMesh:
        """Function that removes duplicated triangles, i.e., removes triangles that reference the same
        three vertices and have the same orientation.
        
        Returns:
            TriangleMesh: Mesh without duplicated triangles.
        """
        ...

    def remove_duplicated_vertices(self) -> TriangleMesh:
        """Function that removes duplicated vertices, i.e., vertices that have identical coordinates.
        
        Returns:
            TriangleMesh: Mesh without duplicated vertices.
        """
        ...

    def remove_non_manifold_edges(self) -> TriangleMesh:
        """Function that removes all non-manifold edges, by successively deleting triangles with the
        smallest surface area adjacent to the non-manifold edge until the number of adjacent triangles
        to the edge is <= 2.
        
        Returns:
            TriangleMesh: Mesh without non-manifold edges.
        """
        ...

    def remove_triangles_by_index(self, triangle_indices: list[int]) -> None:
        """Function that removes the triangles with index in triangle_indices. Call
        remove_unreferenced_vertices to clean up vertices afterwards.
        
        Args:
            triangle_indices (list[int]): 1D array of triangle indices that should
                be removed from the TriangleMesh.
        """
        ...

    def remove_triangles_by_mask(self, triangle_mask: list[bool]) -> None:
        """Function that removes the triangles where triangle_mask is set to true. Call
        remove_unreferenced_vertices to clean up vertices afterwards.
        
        Args:
            triangle_mask (list[bool]): 1D bool array, True values indicate triangles
                that should be removed.
        """
        ...

    def remove_unreferenced_vertices(self) -> TriangleMesh:
        """Function that removes vertices from the triangle mesh that are not referenced
        in any triangle of the mesh.
        
        Returns:
            TriangleMesh: Mesh without unreferenced vertices.
        """
        ...
    
    def remove_vertices_by_index(self, vertex_indices: list[int]) -> None:
        """Function that removes the vertices with index in vertex_indices. Note that also all
        triangles associated with the vertices are removed.
        
        Args:
            vertex_indices (list[int]): 1D array of vertex indices that should be removed from the TriangleMesh.
        """
        ...

    def remove_vertices_by_mask(self, vertex_mask: list[bool]) -> None:
        """Function that removes the vertices that are masked in vertex_mask. Note that also all
        triangles associated with the vertices are removed.
        
        Args:
            vertex_mask (list[bool]): 1D bool array, True values indicate vertices that should be removed.
        """
        ...

    def sample_points_poisson_disk(
        self,
        number_of_points: int,
        init_factor: float = 5,
        pcl: PointCloud = None,
        use_triangle_normal: bool = False,
    ) -> PointCloud:
        """Function to sample points from the mesh, where each point has approximately the same distance
        to the neighbouring points (blue noise). Method is based on Yuksel, “Sample Elimination for Generating
        Poisson Disk Sample Sets”, EUROGRAPHICS, 2015.
        
        Args:
            number_of_points (int): Number of points that should be sampled.
            init_factor (float, optional, default=5): Factor for the initial uniformly sampled PointCloud.
                This init PointCloud is used for sample elimination.
            pcl (PointCloud, optional, default=None): Initial PointCloud that is used for sample elimination.
                If this parameter is provided the init_factor is ignored.
            use_triangle_normal (bool, optional, default=False): If True assigns the triangle normals instead
                of the interpolated vertex normals to the returned points. The triangle normals will be computed
                and added to the mesh if necessary.
        Returns:
            PointCloud: Sampled points.
        """
        ...

    def sample_points_uniformly(
        self,
        number_of_points: int = 100,
        use_triangle_normal: bool = False,
    ) -> PointCloud:
        """Function to uniformly sample points from the mesh.
        
        Args:
            number_of_points (int, optional, default=100): Number of points that should be uniformly sampled.
            use_triangle_normal (bool, optional, default=False): If True assigns the triangle normals instead
                of the interpolated vertex normals to the returned points. The triangle normals will be computed
                and added to the mesh if necessary.
        Returns:
            PointCloud: Sampled points.
        """
        ...

    def select_by_index(self, indices: list[int], cleanup: bool = True) -> TriangleMesh:
        """Function to select mesh from input triangle mesh into output triangle mesh.
        
        Args:
            indices (list[int]): Indices of vertices to be selected.
            cleanup (bool, optional, default=True): If true calls number of mesh cleanup functions to remove
                unreferenced vertices and degenerate triangles
        Returns:
            TriangleMesh: Selected mesh.
        """
        ...

    def simplify_quadric_decimation(
        self,
        target_number_of_triangles: int,
        maximum_error: float = sys.float_info.max,
        boundary_weight: float = 1.0,
    ) -> TriangleMesh:
        """Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert
        
        Args:
            target_number_of_triangles (int): The number of triangles that the simplified mesh should have.
                It is not guaranteed that this number will be reached.
            maximum_error (float, optional, default=inf): The maximum error where a vertex is allowed to be merged
            boundary_weight (float, optional, default=1.0): A weight applied to edge vertices used to preserve boundaries
        Returns:
            TriangleMesh: Simplified mesh.
        """
        ...

    def simplify_vertex_clustering(
        self, voxel_size: float, contraction: SimplificationContraction = SimplificationContraction.Average
    ) -> TriangleMesh:
        """Function to simplify mesh using vertex clustering.
        
        Args:
            voxel_size (float): The size of the voxel within vertices are pooled.

            contraction (SimplificationContraction, optional, default=SimplificationContraction.Average):
                Method to aggregate vertex information. Average computes a simple average, Quadric minimizes
                the distance to the adjacent planes.
        Returns:
            TriangleMesh: Simplified mesh.
        """
        ...

    def subdivide_loop(self, number_of_iterations: int = 1) -> TriangleMesh:
        """Function subdivide mesh using Loop's algorithm. Loop, “Smooth subdivision surfaces based on triangles”, 1987.
        
        Args:
            number_of_iterations (int, optional, default=1): Number of iterations. A single iteration splits
                each triangle into four triangles.
        Returns:
            TriangleMesh: Subdivided mesh.
        """
        ...

    def subdivide_midpoint(self, number_of_iterations: int = 1) -> TriangleMesh:
        """Function subdivide mesh using midpoint algorithm.
        
        Args:
            number_of_iterations (int, optional, default=1): Number of iterations. A single iteration splits
                each triangle into four triangles that cover the same surface.
        Returns:
            TriangleMesh: Subdivided mesh.
        """
        ...

class TetraMesh(MeshBase):
    tetras: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def create_from_point_cloud(cls, point_cloud: PointCloud) -> TetraMesh: ...
    def extract_triangle_mesh(
        self, values: utility.DoubleVector, level: float
    ) -> TriangleMesh: ...
    def has_tetras(self) -> bool: ...
    def remove_degenerate_tetras(self) -> TetraMesh: ...
    def remove_duplicated_tetras(self) -> TetraMesh: ...
    def remove_duplicated_vertices(self) -> TetraMesh: ...
    def remove_unreferenced_vertices(self) -> TetraMesh: ...

class Voxel:
    color: ArrayLike
    grid_index: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...

class VoxelGrid(Geometry3D):
    origin: tuple[float, float, float]
    voxel_size: float
    def __init__(self, *args, **kwargs) -> None: ...
    def __add__(self, voxelgrid: VoxelGrid) -> VoxelGrid: ...
    def __iadd__(self, voxelgrid: VoxelGrid) -> VoxelGrid: ...
    def carve_depth_map(
        self,
        depth_map: Image,
        camera_params: camera.PinholeCameraParameters,
        keep_voxles_outside_image: bool = False,
    ) -> VoxelGrid: ...
    def carve_silhouette(
        self,
        silhouette_mask: Image,
        camera_params: camera.PinholeCameraParameters,
        keep_voxles_outside_image: bool = False,
    ) -> VoxelGrid: ...
    def check_if_included(self, queries: utility.Vector3dVector) -> list[bool]: ...
    @classmethod
    def create_dense(
        cls,
        origin: ArrayLike,
        color: ArrayLike,
        voxel_size: float,
        width: float,
        height: float,
        depth: float,
    ) -> VoxelGrid: ...
    def create_from_octree(self, octree: Octree) -> None: ...
    @classmethod
    def create_from_point_cloud(
        cls, input: PointCloud, voxel_size: float
    ) -> VoxelGrid: ...
    @classmethod
    def create_from_point_cloud_within_bounds(
        cls,
        input: PointCloud,
        voxel_size: float,
        min_bound: ArrayLike,
        max_bound: ArrayLike,
    ) -> VoxelGrid: ...
    @classmethod
    def create_from_triangle_mesh(
        cls, input: TriangleMesh, voxel_size: float
    ) -> VoxelGrid: ...
    @classmethod
    def create_from_triangle_mesh_within_bounds(
        cls,
        input: TriangleMesh,
        voxel_size: float,
        min_bound: ArrayLike,
        max_bound: ArrayLike,
    ) -> VoxelGrid: ...
    def get_voxel(self, point: NDArray[float64]) -> NDArray[float64]: ...
    def get_voxels(self) -> list[Voxel]: ...
    def has_colors(self) -> bool: ...
    def has_voxels(self) -> bool: ...
    def to_octree(self, max_depth: int) -> Octree: ...
    def rotate(
        self, R: NDArray[float64], center: NDArray[float64] = ...
    ) -> VoxelGrid: ...
    def scale(self, scale: float, center: NDArray[float64]) -> VoxelGrid: ...
    def transform(self, transformation: NDArray[float64]) -> VoxelGrid: ...
    def translate(
        self, translation: NDArray[float64], relative: bool = True
    ) -> VoxelGrid: ...

class RGBDImage(Geometry2D):
    color: Image
    depth: Image
    def __init__(self) -> None: ...
    @classmethod
    def create_from_color_and_depth(
        cls,
        color: Image,
        depth: Image,
        depth_scale: float = 1000.0,
        depth_trunc: float = 3.0,
        convert_rgb_to_intensity: bool = True,
    ) -> RGBDImage: ...
    @classmethod
    def create_from_nyu_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...
    @classmethod
    def create_from_redwood_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...
    @classmethod
    def create_from_sun_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...
    @classmethod
    def create_from_tum_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...

class Octree(Geometry3D):
    max_depth: int
    origin: NDArray[float64]
    root_node: OctreeNode
    size: float
    def __init__(self, *args, **kwargs) -> None: ...
    def convert_from_point_cloud(
        self, point_cloud: PointCloud, size_expand: float = 0.01
    ) -> None: ...
    def create_from_voxel_grid(self, voxel_grid: VoxelGrid) -> None: ...
    def insert_point(
        self,
        point: NDArray[float64],
        f_init: Callable[[], OctreeLeafNode],
        f_update: Callable[[OctreeLeafNode], None],
        fi_init: Callable[[], OctreeInternalNode] | None = None,
        fi_update: Callable[[OctreeInternalNode], None] | None = None,
    ) -> None: ...
    @classmethod
    def is_point_in_bound(
        cls, point: NDArray[float64], origin: NDArray[float64], size: float
    ) -> bool: ...
    def locate_leaf_node(
        self, point: NDArray[float64]
    ) -> tuple[OctreeLeafNode, OctreeNodeInfo]: ...
    def to_voxel_grid(self) -> VoxelGrid: ...
    def traverse(self, f: Callable[[OctreeNode, OctreeNodeInfo], bool]) -> None: ...

class OctreeNode:
    def __init__(self, *args, **kwargs) -> None: ...

class OctreeNodeInfo:
    child_index: int
    depth: int
    origin: NDArray[float64]
    size: float
    def __init__(self, *args, **kwargs) -> None: ...

class OctreeLeafNode(OctreeNode):
    def __init__(self, *args, **kwargs) -> None: ...
    def clone(self) -> OctreeLeafNode: ...

class OctreeColorLeafNode(OctreeLeafNode):
    color: NDArray[float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_init_function(cls) -> Callable[[], OctreeLeafNode]: ...
    @classmethod
    def get_update_function(
        cls, color: NDArray[float64]
    ) -> Callable[[OctreeLeafNode], None]: ...

class OctreePointColorLeafNode(OctreeColorLeafNode):
    indices: list[int]
    def __init__(self, *args, **kwargs) -> None: ...

class OctreeInternalNode(OctreeNode):
    children: list[OctreeNode]
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_init_function(cls) -> Callable[[], OctreeInternalNode]: ...
    @classmethod
    def get_update_function(cls) -> Callable[[OctreeInternalNode], None]: ...

class OctreeInternalPointNode(OctreeInternalNode):
    indices: list[int]

class DeformAsRigidAsPossibleEnergy(Enum):
    Smoothed = ...
    Spokes = ...

class FilterScope(Enum):
    All = ...
    Color = ...
    Normal = ...
    Vertex = ...

class SimplificationContraction(Enum):
    Average = ...
    Quadric = ...

class HalfEdge:
    next: int
    triangle_index: int
    twin: int
    vertex_indices: list[int]
    def __init__(self, *args, **kwargs) -> None: ...
    def is_boundary(self) -> bool: ...

class HalfEdgeTriangleMesh(MeshBase):
    half_edges: list[HalfEdge]
    ordered_half_edge_from_vertex: list[list[int]]
    triangle_normals: utility.Vector3dVector
    triangles: utility.Vector3iVector
    def __init__(self, *args, **kwargs) -> None: ...
    def boundary_half_edges_from_vertex(
        self, vertex_index: int
    ) -> utility.IntVector: ...
    def boundary_vertices_from_vertex(self, vertex_index: int) -> utility.IntVector: ...
    @classmethod
    def create_from_triangle_mesh(cls, mesh: TriangleMesh) -> HalfEdgeTriangleMesh: ...
    def has_half_edges(self) -> bool: ...
