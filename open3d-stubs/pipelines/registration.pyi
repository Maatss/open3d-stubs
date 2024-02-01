from typing import Iterable, overload

from numpy import float64, int32
from numpy.typing import NDArray

from .. import geometry, utility

class Feature:
    data: NDArray[float64]
    def __init__(self, *args, **kwargs) -> None: ...
    def dimension(self) -> int: ...
    def num(self) -> int: ...
    def resize(self, dim: int, n: int) -> None: ...

class RobustKernel:
    def __init__(self, *args, **kwargs) -> None: ...
    def weight(self, residual: float) -> float: ...

class CauchyLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...

class GMLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...

class HuberLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...

class L1Loss(RobustKernel):
    def __init__(self, *args, **kwargs) -> None: ...

class L2Loss(RobustKernel):
    def __init__(self, *args, **kwargs) -> None: ...

class TukeyLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...

class CorrespondenceChecker:
    require_pointcloud_alighment_: bool
    def __init__(self, *args, **kwargs) -> None: ...
    def Check(
        self,
        source: geometry.PointCloud,
        target: geometry.PointCloud,
        corres: utility.Vector2iVector,
        transformation: NDArray[float64],
    ) -> bool: ...

class CorrespondenceCheckerBasedOnDistance(CorrespondenceChecker):
    distance_threshold: float
    def __init__(self, *args, **kwargs) -> None: ...

class CorrespondenceCheckerBasedOnEdgeLength(CorrespondenceChecker):
    similarity_threshold: float
    def __init__(self, *args, **kwargs) -> None: ...

class CorrespondenceCheckerBasedOnNormal(CorrespondenceChecker):
    normal_angle_threshold: float
    def __init__(self, *args, **kwargs) -> None: ...

class GlobalOptimizationConvergenceCriteria:
    lower_scale_factor: float
    max_iteration: int
    """Maximum iteration number for iterative optimization module."""
    max_iteration_lm: int 
    """Maximum iteration number for Levenberg Marquardt method.
    max_iteration_lm is used for additional Levenberg-Marquardt
    inner loop that automatically changes steepest gradient gain."""
    min_relative_increment: float
    min_relative_residual_increment: float
    min_residual: float
    min_right_term: float
    upper_scale_factor: float
    """Upper scale factor value. Scaling factors are used for
    levenberg marquardt algorithm these are scaling factors that
    increase/decrease lambda used in H_LM = H + lambda * I"""
    @overload
    def __init__(self) -> None:
        """Default constructor"""
        ...
    @overload
    def __init__(self, other: GlobalOptimizationConvergenceCriteria) -> None:
        """Copy constructor"""
        ...

class GlobalOptimizationOption:
    edge_prune_threshold: float
    """According to [Choi et al 2015], line_process
    weight < edge_prune_threshold (0.25) is pruned."""
    max_correspondence_distance: float
    """Identifies which distance value is used for finding
    neighboring points when making information matrix.
    According to [Choi et al 2015], this distance is used for
    determining $mu, a line process weight."""
    preference_loop_closure: float
    """odometry vs loop-closure. [0,1] -> try to unchange
    odometry edges,[1) -> try to utilize loop-closure. Recommendation:
    0.1 for RGBD Odometry, 2.0 for fragment registration."""
    reference_node: int
    """The pose of this node is unchanged after optimization."""
    @overload
    def __init__(self) -> None:
        """Default constructor"""
        ...

    @overload
    def __init__(self, other: GlobalOptimizationOption) -> None:
        """Copy constructor"""
        ...

    @overload
    def __init__(self, max_correspondence_distance: float = 0.03, edge_prune_threshold: float = 0.25,
                 preference_loop_closure: float = 1.0, reference_node: int = -1) -> None:
        ...
    
    

class GlobalOptimizationMethod:
    """Base class for global optimization method."""

    def __init__(self) -> None: ...

    def optimize_pose_graph(self,
                            pose_graph: PoseGraph,
                            criteria: GlobalOptimizationConvergenceCriteria,
                            option: GlobalOptimizationOption) -> None:
        """Run pose graph optimization."""
        ...

class GlobalOptimizationGaussNewton(GlobalOptimizationMethod):
    """Global optimization with Gauss-Newton algorithm."""

    def __init__(self) -> None:
        """Default constructor."""
        ...

    def __init__(self, other: GlobalOptimizationGaussNewton) -> None:
        """Copy constructor."""
        ...

class GlobalOptimizationLevenbergMarquardt(GlobalOptimizationMethod):
    """
    Global optimization with Levenberg-Marquardt algorithm. Recommended over the Gauss-Newton method since the LM has
    better convergence characteristics.
    """
    
    def __init__(self) -> None:
        """Default constructor."""
        ...

    def __init__(self, other: GlobalOptimizationLevenbergMarquardt) -> None:
        """Copy constructor."""
        ...

class ICPConvergenceCriteria:
    max_iteration: int
    relative_fitness: float
    relative_rmse: float
    @overload
    def __init__(self, other: ICPConvergenceCriteria) -> None: ...
    @overload
    def __init__(
        self,
        relative_fitness: float = 1e-06,
        relative_rmse: float = 1e-06,
        max_iteration: int = 30,
    ) -> None: ...

class PoseGraphEdge:
    confidence: float
    information: NDArray[float64]
    source_node_id: int
    target_node_id: int
    transformation: NDArray[float64]
    uncertain: bool
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: PoseGraphEdge) -> None: ...
    @overload
    def __init__(
        self,
        source_node_id: int = -1,
        target_node_id: int = -1,
        transformation: NDArray[float64] = ...,
        information: NDArray[float64] = ...,
        uncertain: bool = False,
        confidence: float = 1.0,
    ) -> None: ...

class PoseGraphEdgeVector:
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, key) -> PoseGraphEdge: ...
    def __setitem__(self, key, value: PoseGraphEdge) -> PoseGraphEdgeVector: ...
    def append(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def extend(self, L: PoseGraphEdgeVector | Iterable) -> None: ...
    def insert(self, i: int, x: PoseGraphEdge) -> None: ...
    def pop(self, i: int | None) -> PoseGraphEdge: ...

class PoseGraphNode:
    pose: NDArray[float64]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: PoseGraphNode) -> None: ...
    @overload
    def __init__(
        self,
        pose: NDArray[float64],
    ) -> None: ...

class PoseGraphNodeVector:
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, key) -> PoseGraphNode: ...
    def __setitem__(self, key, value: PoseGraphNode) -> PoseGraphNodeVector: ...
    def append(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def extend(self, L: PoseGraphNodeVector | Iterable) -> None: ...
    def insert(self, i: int, x: PoseGraphNode) -> None: ...
    def pop(self, i: int | None) -> PoseGraphNode: ...

class PoseGraph:
    edges: list[PoseGraphEdge]
    nodes: list[PoseGraphNode]
    def __init__(self, *args, **kwargs) -> None: ...

class FastGlobalRegistrationOption:
    decrease_mu: bool
    division_factor: float
    iteration_number: int
    maximum_correspondence_distance: float
    maximum_tuple_count: float
    tuple_scale: float
    use_absolute_scale: bool
    @overload
    def __init__(self, other: FastGlobalRegistrationOption) -> None: ...
    @overload
    def __init__(
        self: FastGlobalRegistrationOption,
        division_factor: float = 1.4,
        use_absolute_scale: bool = False,
        decrease_mu: bool = False,
        maximum_correspondence_distance: float = 0.025,
        iteration_number: int = 64,
        tuple_scale: float = 0.95,
        maximum_tuple_count: int = 1000,
    ) -> None: ...

class RANSACConvergenceCriteria:
    confidence: float
    max_iteration: int
    @overload
    def __init__(self, other: RANSACConvergenceCriteria) -> None: ...
    @overload
    def __init__(
        self, max_iteration: int = 100000, confidence: float = 0.999
    ) -> None: ...

class RegistrationResult:
    correspondence_set: NDArray[int32]
    fitness: float
    inlier_rmse: float
    transformation: NDArray[float64]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: RegistrationResult) -> None: ...

class TransformationEstimation:
    """
    Base class that estimates a transformation between two point clouds. The virtual function ComputeTransformation()
    must be implemented in subclasses.
    """

    def __init__(self, *args, **kwargs) -> None: ...

    def compute_rmse(self, source: geometry.PointCloud, target: geometry.PointCloud,
                     corres: utility.Vector2iVector) -> float:
        """Compute RMSE between source and target points cloud given correspondences.
        
        Args:
            source (geometry.PointCloud): Source point cloud.
            target (geometry.PointCloud): Target point cloud.
            corres (utility.Vector2iVector): Correspondences between source and target point cloud.
        Returns:
            float: RMSE."""
        ...

    def compute_transformation(self, source: geometry.PointCloud, target: geometry.PointCloud,
                               corres: utility.Vector2iVector) -> NDArray[float64]:
        """Compute transformation from source to target point cloud given correspondences.
        
        Args:
            source (geometry.PointCloud): Source point cloud.
            target (geometry.PointCloud): Target point cloud.
            corres (utility.Vector2iVector): Correspondences between source and target point cloud.
        Returns:
            NDArray[float64]: 4x4 transformation matrix."""
        ...

class TransformationEstimationForColoredICP(TransformationEstimation):
    """Transformation estimation for colored point cloud ICP."""

    kernel: RobustKernel
    """Robust kernel used in the optimization."""
    lambda_geometric: float

    @overload
    def __init__(self) -> None:
        """Default constructor."""
        ...
    
    @overload
    def __init__(self, other: TransformationEstimationForColoredICP) -> None:
        """Copy constructor."""
        ...
    
    @overload
    def __init__(self, lambda_geometric: float, kernel: RobustKernel) -> None: ...

    @overload
    def __init__(self, lambda_geometric: float) -> None: ...

    @overload
    def __init__(self, kernel: RobustKernel) -> None: ...

class TransformationEstimationForGeneralizedICP(TransformationEstimation):
    """Class to estimate a transformation for Generalized ICP."""

    epsilon: float
    kernel: RobustKernel
    """Robust kernel used in the optimization."""

    @overload
    def __init__(self) -> None:
        """Default constructor."""
        ...

    @overload
    def __init__(self, other: TransformationEstimationForGeneralizedICP) -> None:
        """Copy constructor."""
        ...

    @overload
    def __init__(self, epsilon: float, kernel: RobustKernel) -> None: ...

    @overload
    def __init__(self, epsilon: float) -> None: ...

    @overload
    def __init__(self, kernel: RobustKernel) -> None: ...

class TransformationEstimationPointToPlane(TransformationEstimation):
    """Class to estimate a transformation for point to plane distance."""

    kernel: RobustKernel
    """Robust kernel used in the optimization."""

    @overload
    def __init__(self) -> None:
        """Default constructor."""
        ...

    @overload
    def __init__(self, other: TransformationEstimationPointToPlane) -> None:
        """Copy constructor."""
        ...

    @overload
    def __init__(self, kernel: RobustKernel) -> None: ...

class TransformationEstimationPointToPoint(TransformationEstimation):
    """Class to estimate a transformation for point to point distance."""

    with_scaling: bool
    """Set to True to estimate scaling, False to force scaling to be 1.
    The homogeneous transformation is given by 
    T = [[cR, t], [0, 1]]
    Sets c=1 if with_scaling is False."""

    @overload
    def __init__(self, other: TransformationEstimationPointToPoint) -> None:
        """Copy constructor."""
        ...

    @overload
    def __init__(self, with_scaling: bool = True) -> None: ...


def compute_fpfh_feature(
    input: geometry.PointCloud,
    search_param: geometry.KDTreeSearchParam
) -> Feature:
    """Function to compute FPFH feature for a point cloud

    Args:
        input (geometry.PointCloud): The input point cloud.
        search_param (geometry.KDTreeSearchParam): KDTree search parameter.
    Returns:
        Feature: The output FPFH feature."""
    ...

def correspondences_from_features(
    source_features: Feature,
    target_features: Feature,
    mutual_filter: bool = False,
    mutual_consistency_ratio: float = 0.1
) -> utility.Vector2iVector:
    """Function to find nearest neighbor correspondences from features

    Args:
        source_features (Feature): The source features stored in (dim, N).
        target_features (Feature): The target features stored in (dim, M).
        mutual_filter (bool = False, optional): Filter correspondences and return the collection of (i, j) s.t. source_features[i] and target_features[j] are mutually the nearest neighbor.
        mutual_consistency_ratio (float = 0.1, optional): Threshold to decide whether the number of filtered correspondences is sufficient. Only used when mutual_filter is enabled.
    Returns:
        utility.Vector2iVector: The output correspondences."""
    ...

def evaluate_registration(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    transformation: NDArray[float64] = None
) -> RegistrationResult:
    """Function for evaluating registration between point clouds.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        transformation (NDArray[float64] = None, optional): The transformation matrix from source to target.
            The 4x4 transformation matrix to transform source to target Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    Returns:
        RegistrationResult: The registration result."""
    ...

def get_information_matrix_from_point_clouds(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    transformation: NDArray[float64] = None
) -> NDArray[float64]:
    """
    Function for computing information matrix from transformation matrix.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        transformation (numpy.ndarray[numpy.float64[4, 4]], optional):
            The 4x4 transformation matrix to transform source to target Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    Returns:
        numpy.ndarray[numpy.float64[6, 6]]
    """
    ...

def global_optimization(
    pose_graph: PoseGraph,
    method: GlobalOptimizationMethod,
    max_correspondence_distance: float,
    transformation: NDArray[float64] = None
) -> NDArray[float64]:
    """
    Function for computing information matrix from transformation matrix.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        transformation (numpy.ndarray[numpy.float64[4, 4]], optional):
            The 4x4 transformation matrix to transform source to target Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    Returns:
        numpy.ndarray[numpy.float64[6, 6]]
    """
    ...

def registration_colored_icp(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    init: NDArray[float64] = None,
    estimation_method: TransformationEstimationForColoredICP = None,
    criteria: ICPConvergenceCriteria = None
) -> RegistrationResult:
    """Function for Colored ICP registration
    
    Args:
        source (geometry.PointCloud): The source point cloud.
        target (geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        init (NDArray[float64], optional, default=numpy.identity(4)): Initial transformation estimation.
        estimation_method (TransformationEstimationForColoredICP, optional,
            default=TransformationEstimationForColoredICP with lambda_geometric=0.968):
            Estimation method. One of `TransformationEstimationPointToPoint`,
            `TransformationEstimationPointToPlane`, `TransformationEstimationForGeneralizedICP`,
            `TransformationEstimationForColoredICP`.
        criteria (ICPConvergenceCriteria, optional, default=ICPConvergenceCriteria with relative_fitness=1e-6,
            relative_rmse=1e-6, max_iteration=30): Convergence criteria.
    Returns:
        RegistrationResult: The registration result."""
    ...

def registration_fgr_based_on_correspondence(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    corres: utility.Vector2iVector,
    option: FastGlobalRegistrationOption = None
) -> RegistrationResult:
    """Function for fast global registration based on a set of correspondences
    
    Args:
        source (geometry.PointCloud): The source point cloud.
        target (geometry.PointCloud): The target point cloud.
        corres (utility.Vector2iVector): o3d.utility.Vector2iVector that stores indices of
            corresponding point or feature arrays.
        option (FastGlobalRegistrationOption, optional, default=FastGlobalRegistrationOption with
            division_factor=tuple_test={} use_absolute_scale=1.4 decrease_mu=False
            maximum_correspondence_distance=true iteration_number=0.025 tuple_scale=64
            maximum_tuple_count=0.95): Fast global registration option.
    Returns:
        RegistrationResult: The registration result."""
    ...

def registration_fgr_based_on_feature_matching(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    source_feature: Feature,
    target_feature: Feature,
    option: FastGlobalRegistrationOption = FastGlobalRegistrationOption()
) -> RegistrationResult:
    """
    Function for global RANSAC registration based on feature matching.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        source_feature (open3d.pipelines.registration.Feature): Source point cloud feature.
        target_feature (open3d.pipelines.registration.Feature): Target point cloud feature.
        option (open3d.pipelines.registration.FastGlobalRegistrationOption, optional):
            Registration option Default value: FastGlobalRegistrationOption class with division_factor= tuple_test={} 
            use_absolute_scale= seed={} decrease_mu=1.4 maximum_correspondence_distance=false iteration_number=true
            tuple_scale=0.025 maximum_tuple_count=64

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...

def registration_generalized_icp(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    init: NDArray[float64] = None,
    estimation_method: TransformationEstimationForGeneralizedICP = None,
    criteria: ICPConvergenceCriteria = None
) -> RegistrationResult:
    """Function for Generalized ICP registration

    Args:
        source (geometry.PointCloud): The source point cloud.
        target (geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        init (NDArray[float64], optional, default=numpy.identity(4)): Initial transformation estimation.
        estimation_method (TransformationEstimationForGeneralizedICP, optional,
            default=TransformationEstimationForGeneralizedICP with epsilon=0.001): Estimation method.
            One of `TransformationEstimationPointToPoint`, `TransformationEstimationPointToPlane`,
            `TransformationEstimationForGeneralizedICP`, `TransformationEstimationForColoredICP`.
        criteria (ICPConvergenceCriteria, optional, default=ICPConvergenceCriteria with relative_fitness=1e-6,
            relative_rmse=1e-6, max_iteration=30): Convergence criteria.
    Returns:
        RegistrationResult: The registration result."""
    ...

def registration_icp(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    init: NDArray[float64] = None,
    estimation_method: TransformationEstimation = TransformationEstimationPointToPoint(),
    criteria: ICPConvergenceCriteria = ICPConvergenceCriteria()
) -> RegistrationResult:
    """
    Function for ICP registration.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        init (numpy.ndarray[numpy.float64[4, 4]], optional):
            Initial transformation estimation Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        estimation_method (open3d.pipelines.registration.TransformationEstimation, optional, default=TransformationEstimationPointToPoint without scaling.): Estimation method. One of (TransformationEstimationPointToPoint, TransformationEstimationPointToPlane, TransformationEstimationForGeneralizedICP, TransformationEstimationForColoredICP)
        criteria (open3d.pipelines.registration.ICPConvergenceCriteria, optional, default=ICPConvergenceCriteria class with relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, and max_iteration=30): Convergence criteria

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...

def registration_ransac_based_on_correspondence(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    corres: utility.Vector2iVector,
    max_correspondence_distance: float,
    estimation_method: TransformationEstimation = None,
    ransac_n: int = 3,
    checkers: list[CorrespondenceChecker] = None,
    criteria: RANSACConvergenceCriteria = None
) -> RegistrationResult:
    """Function for global RANSAC registration based on a set of correspondences

    Args:
        source (geometry.PointCloud): The source point cloud.
        target (geometry.PointCloud): The target point cloud.
        corres (utility.Vector2iVector): o3d.utility.Vector2iVector that stores indices of
            corresponding point or feature arrays.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        estimation_method (TransformationEstimation, optional, default=TransformationEstimationPointToPoint without scaling.): Estimation method.
            One of `TransformationEstimationPointToPoint`, `TransformationEstimationPointToPlane`,
            `TransformationEstimationForGeneralizedICP`, `TransformationEstimationForColoredICP`.
        ransac_n (int, optional, default=3): Fit ransac with `ransac_n` correspondences
        checkers (list[CorrespondenceChecker], optional, default=[]): Vector of checkers to
            check if two point clouds can be aligned. One of `CorrespondenceCheckerBasedOnEdgeLength`,
            `CorrespondenceCheckerBasedOnDistance`, `CorrespondenceCheckerBasedOnNormal`.
        criteria (RANSACConvergenceCriteria, optional, default=RANSACConvergenceCriteria with max_iteration=100000, confidence=0.999): Convergence criteria.
    Returns:
        RegistrationResult: The registration result."""
    ...

def registration_ransac_based_on_feature_matching(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    source_feature: Feature,
    target_feature: Feature,
    mutual_filter: bool,
    max_correspondence_distance: float,
    estimation_method: TransformationEstimation = TransformationEstimationPointToPoint(),
    ransac_n: int = 3,
    checkers: list[CorrespondenceChecker] = [],
    criteria: ICPConvergenceCriteria = ICPConvergenceCriteria()
) -> RegistrationResult:
    """
    Function for global RANSAC registration based on feature matching.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        source_feature (open3d.pipelines.registration.Feature): Source point cloud feature.
        target_feature (open3d.pipelines.registration.Feature): Target point cloud feature.
        mutual_filter (bool): Enables mutual filter such that the correspondence of the source point’s correspondence is itself.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        estimation_method (open3d.pipelines.registration.TransformationEstimation, optional, default=TransformationEstimationPointToPoint without scaling.): Estimation method. One of (TransformationEstimationPointToPoint, TransformationEstimationPointToPlane, TransformationEstimationForGeneralizedICP, TransformationEstimationForColoredICP)
        ransac_n (int, optional, default=3): Fit ransac with ransac_n correspondences
        checkers (List[open3d.pipelines.registration.CorrespondenceChecker], optional, default=[]): Vector of Checker class to check if two point clouds can be aligned. One of (CorrespondenceCheckerBasedOnEdgeLength, CorrespondenceCheckerBasedOnDistance, CorrespondenceCheckerBasedOnNormal)
        criteria (open3d.pipelines.registration.RANSACConvergenceCriteria, optional, default=RANSACConvergenceCriteria class with max_iteration=100000, and confidence=9.990000e-01): Convergence criteria

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...
