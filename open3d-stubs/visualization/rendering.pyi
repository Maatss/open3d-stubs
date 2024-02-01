import enum
from typing import Callable, overload
from numpy import float32
from numpy.typing import NDArray

from .. import geometry
import open3d

class Gradient:
    """Manages a gradient for the unlitGradient shader.In gradient mode, the array of points
    specifies points along the gradient,from 0 to 1 (inclusive). These do need to be evenly
    spaced.Simple greyscale: [ ( 0.0, black ), ( 1.0, white ) ]Rainbow (note the gaps around green):
    [ ( 0.000, blue ), ( 0.125, cornflower blue ), ( 0.250, cyan ), ( 0.500, green ), ( 0.750, yellow ),
    ( 0.875, orange ), ( 1.000, red ) ]The gradient will generate a largish texture, so it should be
    fairly smooth, but the boundaries may not be exactly as specified due to quantization imposed by
    the fixed size of the texture. The points must be sorted from the smallest value to the largest.
    The values must be in the range [0, 1]."""

    class Mode(enum):
        GRADIENT = 0
        LUT = 1

    class Point:
        color: NDArray[float32]
        """[R, G, B, A]. Color values must be in [0.0, 1.0]"""
        value: float
        """Must be within 0.0 and 1.0"""

    mode: Mode
    points: list[Point]

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, points: list[Point]) -> None: ...

class MaterialRecord:
    absorption_color: NDArray[float32]
    absorption_distance: float
    albedo_img: geometry.Image
    anisotropy_img: geometry.Image
    ao_img: geometry.Image
    ao_rough_metal_img: geometry.Image
    aspect_ratio: float
    base_anisotropy: float
    base_clearcoat: float
    base_clearcoat_roughness: float
    base_color: NDArray[float32]
    base_metallic: float
    base_reflectance: float
    base_roughness: float
    clearcoat_img: geometry.Image
    clearcoat_roughness_img: geometry.Image
    generic_imgs: dict[str, geometry.Image]
    generic_params: dict[str, NDArray[float32]]
    gradient: Gradient
    ground_plane_axis: float
    has_alpha: bool
    line_width: float
    metallic_img: geometry.Image
    normal_img: geometry.Image
    point_size: float
    reflectance_img: geometry.Image
    roughness_img: geometry.Image
    sRGB_color: bool
    scalar_max: float
    scalar_min: float
    shader: str
    thickness: float
    transmission: float
    def __init__(self) -> None: ...

class Camera:
    """Camera object"""
    class FovType(enum):
        """Enum class for Camera field of view types."""
        Vertical = 0
        Horizontal = 1

    class Projection(enum):
        """Enum class for Camera projection types."""
        Perspective = 0
        Ortho = 1

    def __init__(*args, **kwargs) -> None: ...


    def copy_from(self, camera: Camera) -> None:
        """Copies the settings from the camera passed as the argument into this camera"""
        ...

    def get_far(self) -> float:
        """Returns the distance from the camera to the far plane"""
        ...

    def get_field_of_view(self) -> float:
        """Returns the field of view of camera, in degrees. Only valid if it was passed to set_projection()."""
        ...

    def get_field_of_view_type(self) -> FovType:
        """Returns the field of view type. Only valid if it was passed to set_projection()."""
        ...

    def get_model_matrix(self) -> NDArray[float32]:
        """Returns the model matrix of the camera

        Returns:
            numpy.ndarray[float32[4, 4]]: The model matrix of the camera
        """
        ...
    
    def get_near(self) -> float:
        """Returns the distance from the camera to the near plane"""
        ...

    def get_projection_matrix(self) -> NDArray[float32]:
        """Returns the projection matrix of the camera

        Returns:
            numpy.ndarray[float32[4, 4]]: The projection matrix of the camera
        """
        ...

    def get_view_matrix(self) -> NDArray[float32]:
        """Returns the view matrix of the camera

        Returns:
            numpy.ndarray[float32[4, 4]]: The view matrix of the camera
        """
        ...

    def look_at(
        self,
        center: NDArray[float32],
        eye: NDArray[float32],
        up: NDArray[float32]
    ) -> None:
        """Sets the position and orientation of the camera

        Args:
            center (numpy.ndarray[float32[3, 1]]): The center of the camera
            eye (numpy.ndarray[float32[3, 1]]): The position of the camera
            up (numpy.ndarray[float32[3, 1]]): The up vector of the camera
        """
        ...

    @overload
    def set_projection(
        self,
        field_of_view: float,
        aspect_ratio: float,
        near_plane: float,
        far_plane: float,
        field_of_view_type: FovType
    ) -> None:
        """Sets a perspective projection."""
        ...
    
    @overload
    def set_projection(
        self,
        projection_type: Projection,
        left: float,
        right: float,
        bottom: float,
        top: float,
        near: float,
        far: float
    ) -> None:
        """Sets the camera projection via a viewing frustum."""
        ...

    @overload
    def set_projection(
        self,
        intrinsics: NDArray[float32],
        near_plane: float,
        far_plane: float,
        image_width: float,
        image_height: float
    ) -> None:
        """Sets the camera projection via intrinsics matrix.

        Args:
            intrinsics (numpy.ndarray[float32[3, 3]]): The intrinsics matrix
            near_plane (float): The distance to the near plane
            far_plane (float): The distance to the far plane
            image_width (float): The width of the image
            image_height (float): The height of the image
        """
        ...

    def unproject(
        self,
        x: float,
        y: float,
        z: float,
        view_width: float,
        view_height: float
    ) -> NDArray[float32]:
        """Takes the (x, y, z) location in the view, where x, y are the number of pixels from the upper left of the view,
        and z is the depth value. Returns the world coordinate (x', y', z').

        Args:
            x (float): The x coordinate
            y (float): The y coordinate
            z (float): The z coordinate
            view_width (float): The width of the view
            view_height (float): The height of the view

        Returns:
            numpy.ndarray[float32[3, 1]]: The world coordinate (x', y', z')
        """
        ...

class Scene:
    """Low-level rendering scene."""

    class GroundPlane(enum):
        """Plane on which to show ground plane: XZ, XY, or YZ."""
        XZ = 0
        XY = 1
        YZ = 2

    UPDATE_POINTS_FLAG: int = 1
    UPDATE_NORMALS_FLAG: int = 2
    UPDATE_COLORS_FLAG: int = 4
    UPDATE_UV0_FLAG: int = 8

    def __init__(*args, **kwargs) -> None: ...

    def add_camera(
        self,
        name: str,
        camera: Camera
    ) -> None:
        """Add a camera to the scene."""
        ...

    def add_directional_light(
        self,
        name: str,
        color: NDArray[float32],
        direction: NDArray[float32],
        intensity: float,
        cast_shadows: bool
    ) -> bool:
        """Adds a directional light to the scene

        Args:
            name (str): The name of the light
            color (numpy.ndarray[float32[3, 1]]): The color of the light
            direction (numpy.ndarray[float32[3, 1]]): The direction of the light
            intensity (float): The intensity of the light
            cast_shadows (bool): Whether the light casts shadows

        Returns:
            bool: True if the light was added successfully, False otherwise
        """
        ...

    @overload
    def add_geometry(
        self,
        name: str,
        geometry: geometry.Geometry3D,
        material: MaterialRecord,
        downsampled_name: str = '',
        downsample_threshold: int = 2^64 - 1
    ) -> bool:
        """Adds a Geometry with a material to the scene"""
        ...

    @overload
    def add_geometry(
        self,
        name: str,
        geometry: open3d.t.geometry.Geometry,
        material: MaterialRecord,
        downsampled_name: str = '',
        downsample_threshold: int = 2^64 - 1
    ) -> bool:
        """Adds a Geometry with a material to the scene"""
        ...

    def add_point_light(
        self,
        name: str,
        color: NDArray[float32],
        position: NDArray[float32],
        intensity: float,
        falloff: float,
        cast_shadows: bool
    ) -> bool:
        """Adds a point light to the scene.

        Args:
            name (str): The name of the light
            color (numpy.ndarray[float32[3, 1]]): The color of the light
            position (numpy.ndarray[float32[3, 1]]): The position of the light
            intensity (float): The intensity of the light
            falloff (float): The falloff of the light
            cast_shadows (bool): Whether the light casts shadows

        Returns:
            bool: True if the light was added successfully, False otherwise
        """
        ...

    def add_spot_light(
        self,
        name: str,
        color: NDArray[float32],
        position: NDArray[float32],
        direction: NDArray[float32],
        intensity: float,
        falloff: float,
        inner_cone_angle: float,
        outer_cone_angle: float,
        cast_shadows: bool
    ) -> bool:
        """Adds a spot light to the scene.

        Args:
            name (str): The name of the light
            color (numpy.ndarray[float32[3, 1]]): The color of the light
            position (numpy.ndarray[float32[3, 1]]): The position of the light
            direction (numpy.ndarray[float32[3, 1]]): The direction of the light
            intensity (float): The intensity of the light
            falloff (float): The falloff of the light
            inner_cone_angle (float): The inner cone angle of the light
            outer_cone_angle (float): The outer cone angle of the light
            cast_shadows (bool): Whether the light casts shadows
        Returns:
            bool: True if the light was added successfully, False otherwise
        """
        ...

    def enable_indirect_light(self, enable: bool) -> None:
        """Enables or disables indirect lighting"""
        ...

    def enable_light_shadow(self, name: str, can_cast_shadows: bool) -> None:
        """Changes whether a point, spot, or directional light can cast shadows."""
        ...
    
    def enable_sun_light(self, enable: bool) -> None:
        """Enables or disables the sun light"""
        ...

    def geometry_is_visible(self, name: str) -> bool:
        """Returns false if the geometry is hidden, True otherwise. Note: this is different from whether or not the geometry is in view."""
        ...

    def geometry_shadows(
        self,
        name: str,
        cast_shadows: bool,
        receive_shadows: bool
    ) -> None:
        """Controls whether an object casts and/or receives shadows: geometry_shadows(name, cast_shadows, receieve_shadows)"""
        ...

    def has_geometry(self, name: str) -> bool:
        """Returns True if a geometry with the provided name exists in the scene."""
        ...

    def remove_camera(self, name: str) -> None:
        """Removes the camera with the given name"""
        ...

    def remove_geometry(self, name: str) -> None:
        """Removes the named geometry from the scene."""
        ...
    
    def remove_light(self, name: str) -> None:
        """Removes the named light from the scene."""
        ...

    def render_to_depth_image(self, callback: Callable[[open3d.geometry.Image], None]) -> None:
        """Renders the scene to a depth image. This can only be used in GUI app.
        To render without a window, use Application.render_to_depth_image. Pixels range from 0.0 (near plane) to 1.0 (far plane)"""
        ...

    def render_to_image(self, callback: Callable[[open3d.geometry.Image], None]) -> None:
        """Renders the scene to an image. This can only be used in a GUI app. To render without a window, use `Application.render_to_image`."""
        ...

    def set_active_camera(self, name: str) -> None:
        """Sets the camera with the given name as the active camera for the scene"""
        ...

    def set_geometry_culling(self, name: str, enable: bool) -> None:
        """Enable/disable view frustum culling on the named object. Culling is enabled by default."""
        ...

    def set_geometry_priority(self, name: str, priority: int) -> None:
        """Set sorting priority for named object. Objects with higher priority will be rendering on top of overlapping geometry with lower priority."""
        ...
    
    def set_indirect_light(self, name: str) -> bool:
        """Loads the indirect light. The name parameter is the name of the file to load"""
        ...

    def set_indirect_light_intensity(self, intensity: float) -> None:
        """Sets the brightness of the indirect light"""
        ...

    def set_sun_light(
        self,
        direction: NDArray[float32],
        color: NDArray[float32],
        intensity: float
    ) -> None:
        """Sets the parameters of the sun light direction, color, intensity

        Args:
            direction (numpy.ndarray[float32[3, 1]]): The direction of the sun light
            color (numpy.ndarray[float32[3, 1]]): The color of the sun light
            intensity (float): The intensity of the sun light
        """
        ...

    def show_geometry(self, name: str, show: bool) -> None:
        """Show or hide the named geometry."""
        ...
    
    def update_geometry(
        self,
        name: str,
        point_cloud: open3d.t.geometry.PointCloud,
        update_flag: int
    ) -> None:
        """Updates the flagged arrays from the t.geometry.PointCloud. The flags should be ORed from `Scene.UPDATE_POINTS_FLAG`,
        `Scene.UPDATE_NORMALS_FLAG`, `Scene.UPDATE_COLORS_FLAG`, and `Scene.UPDATE_UV0_FLAG`"""
        ...
    
    def update_light_color(self, name: str, color: NDArray[float32]) -> None:
        """Changes a point, spot, or directional light's color

        Args:
            name (str): The name of the light
            color (numpy.ndarray[float32[3, 1]]): The color of the light
        """
        ...

    def update_light_cone_angles(
        self,
        name: str,
        inner_cone_angle: float,
        outer_cone_angle: float
    ) -> None:
        """Changes a spot light's inner and outer cone angles."""
        ...

    def update_light_direction(self, name: str, direction: NDArray[float32]) -> None:
        """Changes a spot or directional light's direction.

        Args:
            name (str): The name of the light
            direction (numpy.ndarray[float32[3, 1]]): The direction of the light
        """
        ...

    def update_light_falloff(self, name: str, falloff: float) -> None:
        """Changes a point or spot light's falloff."""
        ...

    def update_light_intensity(self, name: str, intensity: float) -> None:
        """Changes a point, spot or directional light's intensity."""
        ...

    def update_light_position(self, name: str, position: NDArray[float32]) -> None:
        """Changes a point or spot light's position.

        Args:
            name (str): The name of the light
            position (numpy.ndarray[float32[3, 1]]): The position of the light
        """
        ...

class TriangleMeshModel:
    """A list of geometry.TriangleMesh and Material that can describe a complex model
    with multiple meshes, such as might be stored in an FBX, OBJ, or GLTF file"""

    class MeshInfo:
        material_idx: int
        mesh: geometry.TriangleMesh
        mesh_name: str

        def __init__(
            self,
            mesh: geometry.TriangleMesh,
            mesh_name: str,
            material_idx: int
        ) -> None: ...

    materials: list[MaterialRecord]
    meshes: list[MeshInfo]

    def __init__(self) -> None: ...    

class Renderer:
    """Renderer class that manages 3D resources. Get from gui.Window."""

    def __init__(*args, **kwargs) -> None: ...

    def add_texture(
        self,
        image: geometry.Image,
        is_sRGB: bool = False
    ) -> open3d.visualization.rendering.Texture:
        """Adds a texture. The first parameter is the image, the second parameter
        is optional and is True if the image is in the sRGB colorspace and False otherwise"""
        ...

    def remove_texture(self, texture: open3d.visualization.rendering.Texture) -> None:
        """Deletes the texture. This does not remove the texture from any existing materials
        or GUI widgets, and must be done prior to this call."""
        ...

    def set_clear_color(self, color: NDArray[float32]) -> None:
        """Sets the background color for the renderer, [r, g, b, a]. Applies to everything being rendered,
        so it essentially acts as the background color of the window

        Args:
            color (numpy.ndarray[float32[4, 1]]): The color to set
        """
        ...

    def update_texture(
        self,
        texture: open3d.visualization.rendering.Texture,
        image: geometry.Image,
        is_sRGB: bool = False
    ) -> bool:
        """Updates the contents of the texture to be the new image, or returns False and does nothing if the image is a different size.
        It is more efficient to call update_texture() rather than removing and adding a new texture, especially when changes happen frequently,
        such as when implementing video. add_texture(geometry.Image, bool). The first parameter is the image, the second parameter is optional
        and is True if the image is in the sRGB colorspace and False otherwise"""
        ...

class ColorGrading:
    """Parameters to control color grading options"""

    class Quality(enum):
        """Quality level of color grading operations."""
        LOW = 0
        MEDIUM = 1
        HIGH = 2
        ULTRA = 3
    
    class ToneMapping(enum):
        """Specifies the tone-mapping algorithm"""
        LINEAR = 0
        ACES_LEGACY = 1
        ACES = 2
        FILMIC = 3
        UCHIMURA = 4
        REINHARD = 5
        DISPLAY_RANGE = 6

    quality: Quality
    """Quality of color grading operations. High quality is more accurate but slower"""
    temperature: float
    """White balance color temperature"""
    tint: float
    """Tint on the green/magenta axis. Ranges from -1.0 to 1.0."""
    tone_mapping: ToneMapping
    """The tone mapping algorithm to apply. Must be one of `Linear`, `AcesLegacy`, `Aces`, `Filmic`, `Uchimura`, `Rienhard`, `Display Range`(for debug)"""

    def __init__(self, quality: Quality, tone_mapping: ToneMapping) -> None: ...

class View:
    """Low-level view class"""

    class ShadowType(enum):
        """Available shadow mapping algorithm options"""
        PCF = 0
        VSM = 1

    def __init__(*args, **kwargs) -> None: ...

    def get_camera(self) -> Camera:
        """Returns the Camera associated with this View."""
        ...

    def set_ambient_occlusion(self, enabled: bool, ssct_enabled: bool = False) -> None:
        """True to enable, False to disable ambient occlusion. Optionally, screen-space cone tracing may be enabled with ssct_enabled=True."""
        ...

    def set_antialiasing(self, enabled: bool, temporal: bool = False) -> None:
        """True to enable, False to disable anti-aliasing. Note that this only impacts anti-aliasing post-processing.
        MSAA is controlled separately by set_sample_count. Temporal anti-aliasing may be optionally enabled with temporal=True."""
        ...
    
    def set_color_grading(self, color_grading: ColorGrading) -> None:
        """Sets the parameters to be used for the color grading algorithms"""
        ...

    def set_post_processing(self, enabled: bool) -> None:
        """True to enable, False to disable post processing. Post processing effects include: color grading,
        ambient occlusion (and other screen space effects), and anti-aliasing."""
        ...

    def set_sample_count(self, sample_count: int) -> None:
        """Sets the sample count for MSAA. Set to 1 to disable MSAA. Typical values are 2, 4 or 8. The maximum possible value depends on the underlying GPU and OpenGL driver."""
        ...

    def set_shadowing(self, enabled: bool, type: ShadowType = ShadowType.PCF) -> None:
        """True to enable, false to enable all shadow mapping when rendering this View. When enabling shadow mapping
        you may also specify one of two shadow mapping algorithms: `PCF` (default) or `VSM`. Note: shadowing is enabled by default with `PCF` shadow mapping."""
        ...

class Open3DScene:
    """High-level scene for rendering"""

    class LightingProfile(enum):
        """Enum for conveniently setting lighting"""
        HARD_SHADOWS = 0
        DARK_SHADOWS = 1
        MED_SHADOWS = 2
        SOFT_SHADOWS = 3
        NO_SHADOWS = 4

    background_color: NDArray[float32]
    """The background color (read-only)."""
    bounding_box: NDArray[float32]
    """The bounding box of all the items in the scene, visible and invisible (read-only)."""
    camera: Camera
    """The camera object (read-only)."""
    downsample_threshold: int
    """Minimum number of points before downsampled point clouds are created and used when rendering speed is important."""
    scene: Scene
    """The low-level rendering scene object (read-only)."""
    view: View

    def __init__(self, renderer: Renderer) -> None: ...

    @overload
    def add_geometry(
        self,
        name: str,
        geometry: geometry.Geometry3D,
        material: MaterialRecord,
        add_downsampled_copy_for_fast_rendering: bool = True
    ) -> None:
        """Adds a geometry with the specified name. Default visible is true."""
        ...

    @overload
    def add_geometry(
        self,
        name: str,
        geometry: open3d.t.geometry.Geometry,
        material: MaterialRecord,
        add_downsampled_copy_for_fast_rendering: bool = True
    ) -> None:
        """Adds a geometry with the specified name. Default visible is true."""
        ...

    def add_model(self, name: str, model: TriangleMeshModel) -> None:
        """Adds TriangleMeshModel to the scene."""
        ...
    
    def clear_geometry(self) -> None:
        ...

    def geometry_is_visible(self, name: str) -> bool:
        """Returns True if the geometry name is visible"""
        ...

    def get_geometry_transform(self, name: str) -> NDArray[float32]:
        """Returns the pose of the geometry name in the scene.

        Args:
            name (str): The name of the geometry
        Returns:
            numpy.ndarray[float32[4, 4]]: The pose of the geometry name in the scene
        """
        ...

    def has_geometry(self, name: str) -> bool:
        """Returns True if the geometry has been added to the scene, False otherwise"""
        ...

    def modify_geometry_material(self, name: str, material: MaterialRecord) -> None:
        """Modifies the material of the specified geometry"""
        ...

    def remove_geometry(self, name: str) -> None:
        """Removes the geometry with the given name"""
        ...

    def set_background(
        self,
        color: NDArray[float32],
        image: geometry.Image = None
    ) -> None:
        """set_background([r, g, b, a], image=None). Sets the background color and (optionally) image of the scene.

        Args:
            color (numpy.ndarray[float32[4, 1]]): The background color
            image (open3d.geometry.Image, optional): The background image. Defaults to None.
        """
        ...

    def set_background_color(self, color: NDArray[float32]) -> None:
        """This function has been deprecated. Please use set_background() instead.

        Args:
            color (numpy.ndarray[float32[4, 1]]): The background color
        """
        ...

    def set_geometry_transform(self, name: str, transform: NDArray[float32]) -> None:
        """Sets the pose of the geometry name to transform

        Args:
            name (str): The name of the geometry
            transform (numpy.ndarray[float32[4, 4]]): The pose to set
        """
        ...

    def set_lighting(
        self,
        profile: LightingProfile,
        sun_dir: NDArray[float32]
    ) -> None:
        """Sets a simple lighting model. The default value is set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, (0.577, -0.577, -0.577))

        Args:
            profile (LightingProfile): The lighting profile
            sun_dir (numpy.ndarray[float32[3, 1]]): The direction of the sun
        """
        ...

    def set_view_size(self, width: int, height: int) -> None:
        """Sets the view size. This should not be used except for rendering to an image

        Args:
            width (int): The width of the view
            height (int): The height of the view
        """
        ...

    def show_axes(self, enable: bool) -> None:
        """Toggles display of xyz axes"""
        ...

    def show_geometry(self, name: str, show: bool) -> None:
        """Shows or hides the geometry with the given name"""
        ...

    def show_ground_plane(self, enable: bool, plane: Scene.GroundPlane) -> None:
        """Toggles display of ground plane

        Args:
            enable (bool): Whether to enable the ground plane
            plane (Scene.GroundPlane): The plane to display
        """
        ...
    
    def show_skybox(self, enable: bool) -> None:
        """Toggles display of the skybox"""
        ...
    
    def update_material(self, material: MaterialRecord) -> None:
        """Applies the passed material to all the geometries"""
        ...

class OffscreenRenderer:
    """Renderer instance that can be used for rendering to an image"""

    scene: Open3DScene
    """Returns the Open3DScene for this renderer. This scene is destroyed when
    the renderer is destroyed and should not be accessed after that point."""

    def __init__(
        self,
        width: int,
        height: int,
        resource_path: str = ''
    ) -> None:
        """Takes width, height and optionally a resource_path. If unspecified,
        resource_path will use the resource path from the installed Open3D library."""
        ...

    def render_to_depth_image(self, z_in_view_space: bool = False) -> geometry.Image:
        """Renders scene depth buffer to a float image, blocking until the image is returned.
        Pixels range from 0 (near plane) to 1 (far plane). If z_in_view_space is set to True then
        pixels are pre-transformed into view space (i.e., distance from camera)."""
        ...

    def render_to_image(self) -> geometry.Image:
        """Renders scene to an image, blocking until the image is returned"""
        ...

    @overload
    def setup_camera(
        self,
        vertical_field_of_view: float,
        center: NDArray[float32],
        eye: NDArray[float32],
        up: NDArray[float32],
        near_clip: float = -1.0,
        far_clip: float = -1.0
    ) -> None:
        """Sets camera view using bounding box of current geometry if the near_clip and far_clip parameters are not set

        Args:
            vertical_field_of_view (float): The vertical field of view
            center (numpy.ndarray[float32[3, 1]]): The center of the camera
            eye (numpy.ndarray[float32[3, 1]]): The position of the camera
            up (numpy.ndarray[float32[3, 1]]): The up vector of the camera
            near_clip (float, optional): The distance to the near plane. Defaults to -1.0.
            far_clip (float, optional): The distance to the far plane. Defaults to -1.0.
        """
        ...
    
    @overload
    def setup_camera(
        self,
        intrinsics: open3d.camera.PinholeCameraIntrinsic,
        extrinsic_matrix: NDArray[float32]
    ) -> None:
        """Sets the camera view using bounding box of current geometry

        Args:
            intrinsics (open3d.camera.PinholeCameraIntrinsic): The camera intrinsics
            extrinsic_matrix (numpy.ndarray[float32[4, 4]]): The extrinsic matrix
        """
        ...
    
    @overload
    def setup_camera(
        self,
        intrinsic_matrix: NDArray[float32],
        extrinsic_matrix: NDArray[float32],
        intrinsic_width_px: int,
        intrinsic_height_px: int
    ) -> None:
        """Sets the camera view using bounding box of current geometry

        Args:
            intrinsic_matrix (numpy.ndarray[float32[3, 3]]): The intrinsic matrix
            extrinsic_matrix (numpy.ndarray[float32[4, 4]]): The extrinsic matrix
            intrinsic_width_px (int): The width of the intrinsic matrix
            intrinsic_height_px (int): The height of the intrinsic matrix
        """
        ...
