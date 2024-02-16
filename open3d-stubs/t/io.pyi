from __future__ import annotations
from enum import Enum
from typing import Any, overload

import open3d
from open3d import core

class DepthNoiseSimulator:
    """Simulate depth image noise from a given noise distortion model.
    The distortion model is based on Teichman et. al.
    “Unsupervised intrinsic calibration of depth sensors via SLAM” RSS 2009.
    Also see <http://redwood-data.org/indoor/dataset.html>"""

    noise_model: Any

    def __init__(self, noise_model_path: str):
        """Args:
            noise_model_path (str): path to the noise model file.
                See http://redwood-data.org/indoor/dataset.html for the format.
                Or, you may use one of our example datasets, e.g., RedwoodIndoorLivingRoom1.
        """

    def enable_deterministic_debug_mode(self) -> None:
        """Enable deterministic debug mode. All normally distributed noise will be replaced by 0."""

    def simulate(
            self,
            im_src: open3d.t.geometry.Image,
            depth_scale: float = 1000.0
    ) -> open3d.t.geometry.Image:
        """Apply noise model to a depth image.
        
        Args:
            im_src (open3d.t.geometry.Image): Source depth image, must be with
                dtype UInt16 or Float32, channels==1.
            depth_scale (float, optional, default=1000.0): Scale factor to the
                depth image. As a sanity check, if the dtype is Float32,
                the depth_scale must be 1.0. If the dtype is is UInt16,
                the depth_scale is typically larger than 1.0, e.g. it can be 1000.0.
        Returns:
            open3d.t.geometry.Image: Depth image with noise.
        """

class RGBDSensor:
    """Interface class for control of RGBD cameras."""
    def __init__(self, *args, **kwargs) -> None: ...

class RGBDVideoMetadata:
    """RGBD Video metadata."""

    color_channels: int
    """Number of color channels."""
    color_dt: core.Dtype
    """Pixel Dtype for color data."""
    color_format: str
    """Pixel format for color data"""
    depth_dt: core.Dtype
    """Pixel Dtype for depth data."""
    depth_format: str
    """Pixel format for depth data"""
    depth_scale: float
    """Number of depth units per meter (depth in m = depth_pixel_value/depth_scale)."""
    device_name: str
    """Capture device name"""
    fps: float
    """Video frame rate (common for both color and depth)"""
    height: int
    """Height of the video"""
    intrinsics: open3d.camera.PinholeCameraIntrinsic
    """Shared intrinsics between RGB & depth"""
    serial_number: str
    """Capture device serial number"""
    stream_length_usec: int
    """Length of the video (usec). 0 for live capture."""
    width: int
    """Width of the video"""

    def __init__(self) -> None: ...

class RGBDVideoReader:
    """RGBD Video file reader."""

    def __init__(self) -> None: ...

    @staticmethod
    def create(filename: str) -> RGBDVideoReader:
        """Create RGBDVideoReader from filename.

        Args:
            filename (str): RGBD video filename.

        Returns:
            RGBDVideoReader: RGBDVideoReader object.
        """
        ...

    def save_frames(
        self,
        frame_path: str,
        start_time_us: int = 0,
        end_time_us: int = 2^64 - 1
    ) -> None:
        """Save synchronized and aligned individual frames to subfolders.

        Args:
            frame_path (str):  Frames will be stored in stream subfolders
                'color' and 'depth' here. The intrinsic camera calibration
                for the color stream will be saved in 'intrinsic.json'
            start_time_us (int, optional, default=0): Start saving frames from this time (us)
            end_time_us (int, optional, default=2^64 - 1): Save frames until this time (us)
        """
        ...

class RSBagReader:
    """Realsense Bag file reader.

    >Only the first color and depth streams from the bag file will be read.

    - The streams must have the same frame rate.
    - The color stream must have RGB 8 bit (RGB8/BGR8) pixel format
    - The depth stream must have 16 bit unsigned int (Z16) pixel format

    The output is synchronized color and depth frame pairs with the depth frame aligned
    to the color frame. Unsynchronized frames will be dropped. With alignment, the depth
    and color frames have the same viewpoint and resolution. See format documentation 
    at https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html#ae04b7887ce35d16dbd9d2d295d23aac7

    >Warning
    A few frames may be dropped if user code takes a long time (>10 frame intervals) to process a frame.
    """

    metadata: RGBDVideoMetadata
    """Metadata of the RS bag playback."""

    @overload
    def __init__(self) -> None: ... 

    @overload
    def __init__(self, buffer_size: int = 32) -> None:
        """Args:
            buffer_size (int, optional, default=32): Size of internal frame buffer, increase
                this if you experience frame drops."""
        ...
    
    def close(self) -> None:
        """Close the opened RS bag playback."""
        ...
    
    @staticmethod
    def create(filename: str) -> RGBDVideoReader:
        """Create RGBD video reader based on filename

        Args:
            filename (str): Path to the RGBD video file.
        Returns:
            RGBDVideoReader
        """
        ...

    def get_timestamp(self) -> int:
            """Get current timestamp (in us)."""
            ...

    def is_eof(self) -> bool:
        """Check if the RS bag file is all read."""
        ...
    
    def is_opened(self) -> bool:
        """Check if the RS bag file is opened."""
        ...

    def next_frame(self) -> open3d.t.geometry.RGBDImage:
        """Get next frame from the RS bag playback and returns the RGBD object."""

    def open(self, filename: str) -> bool:
            """Open an RS bag playback.

            Args:
                filename (str): Path to the RGBD video file.
            Returns:
                bool
            """
            ...

    def save_frames(
            self,
            frame_path: str,
            start_time_us: int = 0,
            end_time_us: int = 2^64 - 1
    ) -> None:
        """Save synchronized and aligned individual frames to subfolders.

        Args:
            frame_path (str):  Frames will be stored in stream subfolders
                'color' and 'depth' here. The intrinsic camera calibration
                for the color stream will be saved in 'intrinsic.json'
            start_time_us (int, optional, default=0): Start saving frames from this time (us)
            end_time_us (int, optional, default=2^64 - 1): (default video length)
                Save frames until this time (us)
        """
    
    def seek_timestamp(self, timestamp: int) -> bool:
        """Seek to the timestamp (in us).

        Args:
            timestamp (int): Timestamp in the video (usec).
        Returns:
            bool
        """
        ...

class RealSenseSensorConfig:
    """Configuration for a RealSense camera."""

    @overload
    def __init__(self) -> None:
        """Default config will be used."""
        ...

    @overload
    def __init__(self, config: dict[str, str]) -> None:
        """Initialize config with a map."""
        ...

class RealSenseValidConfigs:
    """Store set of valid configuration options for a connected RealSense device.
    From this structure, a user can construct a RealSenseSensorConfig object meeting
    their specifications."""

    name: str
    """Device name"""
    serial: str
    """Device serial number"""
    valid_configs: dict[str, list[Any]]
    """Mapping between configuration option name and a list of valid values."""



class RealSenseSensor:
    """RealSense camera discovery, configuration, streaming and recording"""

    def __init__(self) -> None:
        """Initialize with default settings."""
        ...

    def capture_frame(
        self,
        wait: bool = True,
        align_depth_to_color: bool = True
    ) -> open3d.t.geometry.RGBDImage:
        """Acquire the next synchronized RGBD frameset from the camera.

        Args:
            wait (bool, optional, default=True): If true wait for the next frame set,
                else return immediately with an empty RGBDImage if it is not yet available.
            align_depth_to_color (bool, optional, default=True): Enable aligning WFOV
                depth image to the color image in visualizer.
        Returns:
            open3d.t.geometry.RGBDImage
        """
        ...

    @staticmethod
    def enumerate_devices() -> list[RealSenseValidConfigs]:
        """Query all connected RealSense cameras for their capabilities."""
        ...
    
    def get_filename(self) -> str:
        """Get filename being written."""
        ...

    def get_metadata(self) -> RGBDVideoMetadata:
        """Get metadata of the RealSense video capture."""
        ...

    def get_timestamp(self) -> int:
        """Get current timestamp (in us)"""
        ...

    def init_sensor(
            self,
            sensor_config: RealSenseSensorConfig,
            sensor_index: int,
            filename: str
    ) -> bool:
        """Configure sensor with custom settings. If this is skipped, default settings will be used. You can enable recording to a bag file by specifying a filename.

        Args:
            sensor_config (RealSenseSensorConfig, optional) - Camera configuration,
                such as resolution and framerate. A serial number can be entered here
                to connect to a specific camera.
            sensor_index (int, optional, default=0) - Connect to a camera at this
                position in the enumeration of RealSense cameras that are
                currently connected. Use enumerate_devices() or list_devices()
                to obtain a list of connected cameras. This is ignored if
                sensor_config contains a serial entry.
            filename (str, optional, default='') - Save frames to a bag file
        Returns:
            bool
        """
        ...
    
    @staticmethod
    def list_devices() -> bool:
        """List all RealSense cameras connected to the system along with their
        capabilities. Use this listing to select an appropriate configuration for a camera."""
        ...

    def pause_record(self) -> None:
        """Pause recording to the bag file. Note: If this is called immediately after
        start_capture, the bag file may have an incorrect end time."""
        ...

    def resume_record(self) -> None:
        """Resume recording to the bag file. The file will contain discontinuous segments."""
        ...

    def start_capture(self, start_record: bool = False) -> bool:
        """Start capturing synchronized depth and color frames.

        Args:
            start_record (bool, optional, default=False): Start recording to the specified bag file as well.
        Returns:
            bool
        """
        ...

    def stop_capture(self) -> None:
        """Stop capturing frames."""
        ...

class SensorType(Enum):
    """Sensor type"""
    AZURE_KINECT = 0
    REAL_SENSE = 1


def read_image(filename: str) -> open3d.t.geometry.Image:
    """Function to read image from file.

    Args:
        filename (str): Image file name.
    Returns:
        open3d.t.geometry.Image
    """
    ...

def read_point_cloud(
    filename: str,
    format: str = "auto",
    remove_nan_points: bool = False,
    remove_infinite_points: bool = False,
    print_progress: bool = False
) -> open3d.t.geometry.PointCloud:
    """Function to read point cloud with tensor attributes from file.

    Args:
        filename (str): Path to file.
        format (str, optional, default='auto'): The format of the input file.
            When not specified or set as auto, the format is inferred from
            file extension name.
        remove_nan_points (bool, optional, default=False): If true, all points
            that include a NaN are removed from the PointCloud.
        remove_infinite_points (bool, optional, default=False): If true, all
            points that include an infinite value are removed from the PointCloud.
        print_progress (bool, optional, default=False): If set to true a progress
            bar is visualized in the console.
    Returns:
        open3d.t.geometry.PointCloud
    """
    ...

def read_triangle_mesh(
    filename: str,
    enable_post_processing: bool = False,
    print_progress: bool = False
) -> open3d.t.geometry.TriangleMesh:
    """The general entrance for reading a TriangleMesh from a file. The function
    calls read functions based on the extension name of filename. Supported
    formats are obj, ply, stl, off, gltf, glb, fbx.

    The following example reads a triangle mesh with the .ply extension::
    ```
    import open3d as o3d
    mesh = o3d.t.io.read_triangle_mesh('mesh.ply')
    ```

    Args:
        filename (str): Path to the mesh file.
        enable_post_processing (bool, optional, default=False): If True enables
            post-processing. Post-processing will:

            - triangulate meshes with polygonal faces
            - remove redundant materials
            - pretransform vertices
            - generate face normals if needed

            For more information see ASSIMP's documentation on the flags
            aiProcessPreset_TargetRealtime_Fast, aiProcess_RemoveRedundantMaterials,
            aiProcess_OptimizeMeshes, aiProcess_PreTransformVertices.
            Note that identical vertices will always be joined regardless of
            whether post-processing is enabled or not, which changes the number
            of vertices in the mesh.
            The ply-format is not affected by the post-processing.
        print_progress (bool, optional, default=False): If True print the reading
            progress to the terminal.
    Returns:
        open3d.t.geometry.TriangleMesh: Returns the mesh object. On failure an
            empty mesh is returned.
    """
    ...

def write_image(
    filename: str,
    image: open3d.t.geometry.Image,
    quality: int = -1
) -> bool:
    """Function to write image to file.

    Args:
        filename (str): Path to file.
        image (open3d.t.geometry.Image): The Image object for I/O.
        quality (int, optional, default=-1): Quality of the output file.
    Returns:
        bool
    """
    ...

def write_point_cloud(
    filename: str,
    write_ascii: bool = False,
    compressed: bool = False,
    print_progress: bool = False
) -> bool:
    """Function to write PointCloud with tensor attributes to file.

    Args:
        filename (str): Path to file.
        pointcloud (open3d.t.geometry.PointCloud): The PointCloud object for I/O.
        write_ascii (bool, optional, default=False): Set to True to output in
            ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to True to write in compressed format.
        print_progress (bool, optional, default=False): If set to true a progress
            bar is visualized in the console.
    Returns:
        bool
    """
    ...

def write_triangle_mesh(
    filename: str,
    mesh: open3d.t.geometry.TriangleMesh,
    write_ascii: bool = False,
    compressed: bool = False,
    write_vertex_normals: bool = True,
    write_vertex_colors: bool = True,
    write_triangle_uvs: bool = True,
    print_progress: bool = False
) -> bool:
    """Function to write TriangleMesh to file.

    Args:
        filename (str): Path to file.
        mesh (open3d.t.geometry.TriangleMesh): The TriangleMesh object for I/O.
        write_ascii (bool, optional, default=False): Set to True to output in
            ascii format, otherwise binary format will be used.
        compressed (bool, optional, default=False): Set to True to write in compressed format.
        write_vertex_normals (bool, optional, default=True): Set to False to not
            write any vertex normals, even if present on the mesh.
        write_vertex_colors (bool, optional, default=True): Set to False to not
            write any vertex colors, even if present on the mesh.
        write_triangle_uvs (bool, optional, default=True): Set to False to not
            write any triangle uvs, even if present on the mesh. For obj format,
            mtl file is saved only when True is set.
        print_progress (bool, optional, default=False): If set to true a progress
            bar is visualized in the console.
    Returns:
        bool
    """
    ...
