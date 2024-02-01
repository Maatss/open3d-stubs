import enum
from typing import Callable, overload

import numpy as np

import open3d


class FontStyle(enum):
    """Font style"""
    NORMAL: int = 0
    BOLD: int = 1
    ITALIC: int = 2
    BOLD_ITALIC: int = 3

class FontDescription:
    """Class to describe a custom font"""

    MONOSPACE: str = "monospace"
    SANS_SERIF: str = "sans-serif"

    def __init__(
        self,
        typeface: str = SANS_SERIF,
        style: FontStyle = FontStyle.NORMAL,
        point_size: int = 0,
    ) -> None:
        """Creates a FontDescription.

        'typeface' is a path to a TrueType (.ttf), TrueType Collection (.ttc),
        or OpenType (.otf) file, or it is the name of the font, in which case
        the system font paths will be searched to find the font file. This
        typeface will be used for roman characters (Extended Latin, that is,
        European languages.
        """
        ...

    def add_typeface_for_code_points(self, typeface: str, code_points: list[int]) -> None:
        """Adds specific code points from the typeface.

        This is useful for selectively adding glyphs, for example, from an icon
        font.
        """
        ...

    def add_typeface_for_language(self, typeface: str, language: str) -> None:
        """Adds code points outside Extended Latin from the specified typeface.

        Supported languages are:
        'ja' (Japanese)
        'ko' (Korean)
        'th' (Thai)
        'vi' (Vietnamese)
        'zh' (Chinese, 2500 most common characters, 50 MB per window)
        'zh_all' (Chinese, all characters, ~200 MB per window)

        All other languages will be assumed to be Cyrillic. Note that generally
        fonts do not have CJK glyphs unless they are specifically a CJK font,
        although operating systems generally use a CJK font for you. We do not
        have the information necessary to do this, so you will need to provide
        a font that has the glyphs you need. In particular, common fonts like
        'Arial', 'Helvetica', and SANS_SERIF do not contain CJK glyphs.
        """
        ...

class Menu:
    """A menu, possibly a menu tree"""

    def __init__(self) -> None:
        ...
    
    def add_item(self, label: str, id: int) -> None:
        """Adds a menu item with id to the menu"""
        ...

    def add_menu(self, label: str, menu: Menu) -> None:
        """Adds a submenu to the menu"""
        ...

    def  add_separator(self) -> None:
        """Adds a separator to the menu"""
        ...

    def is_checked(self, id: int) -> bool:
        """Returns True if menu item is checked"""
        ...

    def set_checked(self, id: int, checked: bool) -> None:
        """Sets menu item (un)checked"""
        ...
    
    def set_enabled(self, id: int, enabled: bool) -> None:
        """Sets menu item enabled or disabled"""
        ...

class Rect:
    """Represents a widget frame"""

    height: int
    width: int
    x: int
    y: int

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None: ...
    
    @overload
    def __init__(self, arg0: float, arg1: float, arg2: float, arg3: float) -> None: ...

    def get_bottom(self) -> int:
        """Returns the bottom of the rect"""
        ...
    
    def get_left(self) -> int:
        """Returns the left of the rect"""
        ...
    
    def get_right(self) -> int:
        """Returns the right of the rect"""
        ...
    
    def get_top(self) -> int:
        """Returns the top of the rect"""
        ...

class Theme:
    """Theme parameters such as colors used for drawing widgets"""

    default_layout_spacing: float
    """Good value for the spacing parameter in layouts (read-only)"""
    default_margin: float
    """Good default value for margins, useful for layouts (read-only)"""
    font_size: float
    """Font size (which is also the conventional size of the em unit) (read-only)"""

    def __init__(self, *args, **kwargs) -> None:
        ...

class LayoutContext:
    """Context passed to Window's on_layout callback"""
    theme: Theme

    def __init__(self) -> None: ...

class Size:
    """Size object"""

    height: float
    width: float

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: int) -> None: ...

    @overload
    def __init__(self, arg0: float, arg1: float) -> None: ...

class Color:
    """Stores color for gui classes"""

    alpha: float
    """Returns alpha channel in the range [0.0, 1.0] (read-only)"""
    blue: float
    """Returns blue channel in the range [0.0, 1.0] (read-only)"""
    green: float
    """Returns green channel in the range [0.0, 1.0] (read-only)"""
    red: float
    """Returns red channel in the range [0.0, 1.0] (read-only)"""

    def __init__(self, r: float = 1.0, g: float = 1.0, b: float = 1.0, a: float = 1.0) -> None:
        ...
    
    def set_color(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Sets red, green, blue, and alpha channels, (range: [0.0, 1.0])"""
        ...

class Widget:
    """Base widget class"""

    class Constraints:
        """Constraints object for Widget.calc_preferred_size()"""

        height: float
        """Height constraint"""
        width: float
        """Width constraint"""

        def __init__(self) -> None:
            ...
    
    class EventCallbackResult(enum):
        """Returned by event handlers"""
        IGNORED: int = 0
        """Event handler ignored the event, widget will handle event normally"""
        HANDLED: int = 1
        """Event handler handled the event, but widget will still handle the event normally.
        This is useful when you are augmenting base functionality"""
        CONSUMED: int = 2
        """Event handler consumed the event, event handling stops, widget will not handle the event.
        This is useful when you are replacing functionality"""

    background_color: Color
    """Background color of the widget"""
    enabled: bool
    """True if widget is enabled, False if disabled"""
    frame: Rect
    """The widget's frame. Setting this value will be overridden if the frame is within a layout."""
    tooltip: str
    """Widget's tooltip that is displayed on mouseover"""
    visible: bool
    """True if widget is visible, False otherwise"""

    def __init__(self) -> None: ...

    def add_child(self, widget: Widget) -> None:
        """Adds a child widget"""
        ...

    def calc_preferred_size(
            self,
            ctx: LayoutContext,
            constraints: Constraints
    ) -> Size:
        """Returns the preferred size of the widget.

        This is intended to be called only during layout, although it will also
        work during drawing. Calling it at other times will not work, as it
        requires some internal setup in order to function properly.
        """
        ...

    def get_children(self) -> list[Widget]:
        """Returns the array of children. Do not modify."""
        ...

class KeyName(enum):
    A = 97
    ALT = 260
    AMPERSAND = 38
    ASTERISK = 42
    AT = 64
    B = 98
    BACKSLASH = 92
    BACKSPACE = 8
    BACKTICK = 96
    C = 99
    CAPS_LOCK = 262
    CARET = 94
    COLON = 58
    COMMA = 44
    D = 100
    DELETE = 127
    DOLLAR_SIGN = 36
    DOUBLE_QUOTE = 34
    DOWN = 266
    E = 101
    EIGHT = 56
    END = 269
    ENTER = 10
    EQUALS = 61
    ESCAPE = 27
    EXCLAMATION_MARK = 33
    F = 102
    F1 = 290
    F10 = 299
    F11 = 300
    F12 = 301
    F2 = 291
    F3 = 292
    F4 = 293
    F5 = 294
    F6 = 295
    F7 = 296
    F8 = 297
    F9 = 298
    FIVE = 53
    FOUR = 52
    G = 103
    GREATER_THAN = 62
    H = 104
    HASH = 35
    HOME = 268
    I = 105
    INSERT = 267
    J = 106
    K = 107
    L = 108
    LEFT = 263
    LEFT_BRACE = 123
    LEFT_BRACKET = 91
    LEFT_CONTROL = 258
    LEFT_PAREN = 40
    LEFT_SHIFT = 256
    LESS_THAN = 60
    M = 109
    META = 261
    MINUS = 45
    N = 110
    NINE = 57
    NONE = 0
    O = 111
    ONE = 49
    P = 112
    PAGE_DOWN = 271
    PAGE_UP = 270
    PERCENT = 37
    PERIOD = 46
    PIPE = 124
    PLUS = 43
    Q = 113
    QUESTION_MARK = 63
    QUOTE = 39
    R = 114
    RIGHT = 264
    RIGHT_BRACE = 125
    RIGHT_BRACKET = 93
    RIGHT_CONTROL = 259
    RIGHT_PAREN = 41
    RIGHT_SHIFT = 257
    S = 115
    SEMICOLON = 59
    SEVEN = 55
    SIX = 54
    SLASH = 47
    SPACE = 32
    T = 116
    TAB = 9
    THREE = 51
    TILDE = 126
    TWO = 50
    U = 117
    UNDERSCORE = 95
    UNKNOWN = 1000
    UP = 265
    V = 118
    W = 119
    X = 120
    Y = 121
    Z = 122
    ZERO = 48

class KeyModifier(enum):
    """Key modifier identifiers."""
    NONE = 0
    SHIFT = 1
    CTRL = 2
    ALT = 4
    META = 8

class KeyEvent:
    """Object that stores key events"""

    class Type(enum):
        DOWN: int = 0
        UP: int = 1
    
    is_repeat: bool
    """True if this key down event comes from a key repeat"""
    key: int
    """This is the actual key that was pressed, not the character
    generated by the key.This event is not suitable for text entry"""
    type: Type

    def __init__(*args, **kwargs) -> None: ...

class MouseButton(enum):
    """Mouse button identifiers."""
    NONE = 0
    LEFT = 1
    MIDDLE = 2
    RIGHT = 4
    BUTTON4 = 8
    BUTTON5 = 16

class MouseEvent:
    """Object that stores mouse events"""

    class Type(enum):
        MOVE: int = 0
        BUTTON_DOWN: int = 1
        DRAG: int = 2
        BUTTON_UP: int = 3
        WHEEL: int = 4

    buttons: list[MouseButton]
    """ORed mouse buttons"""
    modifiers: list[KeyModifier]
    """ORed mouse modifiers"""
    type: Type
    """Mouse event type"""
    wheel_dx: float
    """Mouse wheel horizontal motion"""
    wheel_dy: float
    """Mouse wheel vertical motion"""
    wheel_is_trackpad: bool
    """Is mouse wheel event from a trackpad"""
    x: float
    """x coordinate of the mouse event"""
    y: float
    """y coordinate of the mouse event"""

    def __init__(*args, **kwargs) -> None: ...

    def is_button_down(self, button: MouseButton) -> bool:
        """Convenience function to more easily deterimine if a mouse button is pressed"""
        ...

    def is_modifier_down(self, modifier: KeyModifier) -> bool:
        """Convenience function to more easily deterimine if a modifier key is down"""
        ...

class Dialog(Widget):

    def __init__(self, title: str) -> None:
        """Creates a dialog with the given title"""
        ...

class FileDialog(Widget):
    """File picker dialog"""

    class Mode(enum):
        """Enum class for FileDialog modes."""
        OPEN = 0
        SAVE = 1
        OPEN_DIR = 2

    def __init__(
        self,
        mode: Mode,
        title: str,
        theme: Theme,
    ) -> None:
        """Creates either an open or save file dialog.

        The first parameter is either `FileDialog.OPEN` or `FileDialog.SAVE`.
        The second is the title of the dialog, and the third is the theme,
        which is used internally by the dialog for layout. The theme should
        normally be retrieved from `window.theme`.
        """
        ...
    
    def add_filter(self, extension: str, description: str) -> None:
        """Adds a selectable file-type filter: add_filter('.stl', 'Stereolithography mesh')"""
        ...
    
    def set_on_cancel(self, callback: Callable[[], None]) -> None:
        """Cancel callback; required"""
        ...
    
    def set_on_done(self, callback: Callable[[str], None]) -> None:
        """Done callback; required"""
        ...
    
    def set_path(self, path: str) -> None:
        """Sets the initial path path of the dialog"""
        ...

class WindowBase:
    """Application window"""
    def __init__(*args, **kwargs) -> None: ...

class Window(WindowBase):
    """Application window. Create with Application.instance.create_window()."""

    content_rect: Rect
    """Returns the frame in device pixels, relative to the window, which is available for widgets (read-only)"""
    is_active_window: bool
    """True if the window is currently the active window (read-only)"""
    is_visible: bool
    """True if window is visible (read-only)"""
    os_frame: Rect
    """Window rect in OS coords, not device pixels"""
    renderer: open3d.visualization.rendering.Renderer
    """Gets the rendering.Renderer object for the Window"""
    scaling: float
    """Returns the scaling factor between OS pixels and device pixels (read-only)"""
    size: tuple[int, int]
    """The size of the window in device pixels, including menubar (except on macOS)"""
    theme: Theme
    """Get's window's theme info"""
    title: str
    """Returns the title of the window"""

    def __init__(self, *args, **kwargs) -> None: ...

    def add_child(self, widget: Widget) -> None:
        """Adds a widget to the window"""
        ...

    def close(self) -> None:
        """Closes the window and destroys it, unless an on_close callback cancels the close."""
        ...
    
    def close_dialog(self) -> None:
        """Closes the current dialog"""
        ...
    
    def post_redraw(self) -> None:
        """Sends a redraw message to the OS message queue"""
        ...
    
    def set_focus_widget(self, widget: Widget) -> None:
        """Makes specified widget have text focus"""
        ...
    
    def set_needs_layout(self) -> None:
        """Flags window to re-layout"""
        ...
    
    def set_on_close(self, callback: Callable[[], bool]) -> None:
        """Sets a callback that will be called when the window is closed.

        The callback is given no arguments and should return True to continue
        closing the window or False to cancel the close
        """
        ...

    def set_on_key(self, callback: Callable[[KeyEvent], bool]) -> None:
        """Sets a callback for key events.

        This callback is passed a KeyEvent object. The callback must return
        True to stop more dispatching or False to dispatch to focused widget
        """
        ...

    def set_on_layout(self, callback: Callable[[LayoutContext], None]) -> None:
        """Sets a callback function that manually sets the frames of children of the window.

        Callback function will be called with one argument: gui.LayoutContext
        """
        ...
    
    def set_on_menu_item_activated(self, id: int, callback: Callable[[], None]) -> None:
        """Sets callback function for menu item: callback()"""
        ...

    def set_on_tick_event(self, callback: Callable[[], bool]) -> None:
        """Sets callback for tick event.

        Callback takes no arguments and must return True if a redraw is needed
        (that is, if any widget has changed in any fashion) or False if nothing
        has changed
        """
        ...
    
    def show(self, show: bool) -> None:
        """Shows or hides the window"""
        ...
    
    def show_dialog(self, dialog: Dialog) -> None:
        """Displays the dialog"""
        ...

    def show_menu(self, show: bool) -> None:
        """show_menu(show): shows or hides the menu in the window, except on macOS
        since the menubar is not in the window and all applications must have a menubar."""
        ...

    def show_message_box(self, title: str, message: str) -> None:
        """Displays a simple dialog with a title and message and okay button"""
        ...
    
    def size_to_fit(self) -> None:
        """Sets the width and height of window to its preferred size"""
        ...

class Application:
    """Global application singleton. This owns the menubar, windows, and event loop"""

    DEFAULT_FONT_ID: int = 0

    instance: Application
    """Application singleton instance"""
    menubar: Menu
    """The Menu for the application (initially None)"""
    now: float
    """Returns current time in seconds"""
    resource_path: str
    """Returns a string with the path to the resources directory"""

    def __init__(self, *args, **kwargs) -> None:
        ...

    def add_font(self, font: FontDescription) -> int:
        """Adds a font.

        Must be called after initialize() and before a window is created.
        Returns the font id, which can be used to change the font in widgets
        such as Label which support custom fonts.
        """
        ...
    
    def add_window(self, window: Window) -> None:
        """Adds a window to the application.

        This is only necessary when creating an object that is a Window
        directly, rather than with create_window
        """
        ...
    
    def create_window(
        self,
        title: str = "",
        width: int = -1,
        height: int = -1,
        x: int = -1,
        y: int = -1,
        flags: int = 0,
    ) -> Window:
        """Creates a window and adds it to the application.

        To programmatically destroy the window do window.close().
        
        Usage:
        create_window(title, width, height, x, y, flags). x, y, and flags are
        optional.
        """
        ...

    @overload
    def initialize(self) -> None:
        """Initializes the application, using the resources included in the wheel.
        One of the initialize functions _must_ be called prior to using anything in the gui module
        """
        ...

    @overload
    def initialize(self, resource_path: str) -> None:
        """Initializes the application with location of the resources provided by the caller.
        One of the initialize functions _must_ be called prior to using anything in the gui module
        """
        ...

    def post_to_main_thread(
        self,
        window: Window,
        callback: Callable[[],
        None]
    ) -> None:
        """Runs the provided function on the main thread.

        This can be used to execute UI-related code at a safe point in time.
        If the UI changes, you will need to manually request a redraw of the
        window with w.post_redraw()
        """
        ...

    def quit(self) -> None:
        """Closes all the windows, exiting as a result"""
        ...
    
    def render_to_image(
        self,
        scene: open3d.visualization.rendering.Open3DScene,
        width: int,
        height: int,
    ) -> open3d.geometry.Image:
        """Renders a scene to an image and returns the image.

        If you are rendering without a visible window you should use
        open3d.visualization.rendering.RenderToImage instead
        """
        ...
    
    def run(self) -> None:
        """Runs the event loop.

        After this finishes, all windows and widgets should be considered
        uninitialized, even if they are still held by Python variables.
        Using them is unsafe, even if run() is called again.
        """
        ...
    
    def run_in_thread(self, callback: Callable[[], None]) -> None:
        """Runs function in a separate thread.

        Do not call GUI functions on this thread, call post_to_main_thread()
        if this thread needs to change the GUI.
        """
        ...
    
    def run_one_tick(self) -> bool:
        """Runs the event loop once, returns True if the app is still running,
        or False if all the windows have closed or quit() has been called."""
        ...
    
    def set_font(self, id: int, font: FontDescription) -> None:
        """Changes the contents of an existing font, for instance, the default font."""
        ...

class Margins:
    """Margins for layouts."""

    bottom: float
    left: float
    right: float
    top: float

    @overload
    def __init__(
        self,
        left: int = 0,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
    ) -> None:
        """Creates margins. Arguments are left, top, right, bottom. Use the em-size
        (window.theme.font_size) rather than pixels for more consistency across
        platforms and monitors. Margins are the spacing from the edge of the
        widget's frame to its content area. They act similar to the 'padding'
        property in CSS"""
        ...
    
    @overload
    def __init__(
        self,
        left: float = 0.0,
        top: float = 0.0,
        right: float = 0.0,
        bottom: float = 0.0,
    ) -> None:
        """Creates margins. Arguments are left, top, right, bottom. Use the em-size
        (window.theme.font_size) rather than pixels for more consistency across
        platforms and monitors. Margins are the spacing from the edge of the
        widget's frame to its content area. They act similar to the 'padding'
        property in CSS"""
        ...

    def get_horiz(self) -> int: ...
    def get_vert(self) -> int: ...

class Layout1D(Widget):

    def __init__(self, *args, **kwargs) -> None:
        ...

    @overload
    def add_fixed(self, size: int) -> None:
        """Adds a fixed amount of empty space to the layout"""
        ...
    
    @overload
    def add_fixed(self, size: float) -> None:
        """Adds a fixed amount of empty space to the layout"""
        ...
    
    def add_stretch(self) -> None:
        """Adds empty space to the layout that will take up as much extra space as there is available in the layout"""
        ...

class Horiz(Layout1D):
    """Horizontal layout"""

    @overload
    def __init__(
        self,
        spacing: int = 0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets horizontally, left to right,
        making their height equal to the layout's height (which will generally
        be the largest height of the items). First argument is the spacing between
        widgets, the second is the margins. Both default to 0."""
        ...
    
    @overload
    def __init__(
        self,
        spacing: float = 0.0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets horizontally, left to right,
        making their height equal to the layout's height (which will generally
        be the largest height of the items). First argument is the spacing between
        widgets, the second is the margins. Both default to 0."""
        ...

class Vert(Layout1D):
    """Vertical layout."""

    @overload
    def __init__(
        self,
        spacing: int = 0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets vertically, top to bottom,
        making their width equal to the layout's width. First argument is the
        spacing between widgets, the second is the margins. Both default to 0.
        """
        ...
    
    @overload
    def __init__(
        self,
        spacing: float = 0.0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets vertically, top to bottom,
        making their width equal to the layout's width. First argument is the
        spacing between widgets, the second is the margins. Both default to 0.
        """
        ...

class CollapsableVert(Layout1D):
    """Vertical layout with title, whose contents are collapsable"""

    font_id: int
    """Set the font using the FontId returned from Application.add_font()"""

    @overload
    def __init__(
        self,
        text: str,
        spacing: int = 0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets vertically, top to bottom,
        making their width equal to the layout's width. First argument is the
        heading text, the second is the spacing between widgets, and the third
        is the margins. Both the spacing and the margins default to 0.
        """
        ...

    @overload
    def __init__(
        self,
        text: str,
        spacing: float = 0.0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets vertically, top to bottom,
        making their width equal to the layout's width. First argument is the
        heading text, the second is the spacing between widgets, and the third
        is the margins. Both the spacing and the margins default to 0.
        """
        ...
    
    def get_is_open(self) -> bool:
        """Check if widget is open."""
        ...

    def set_is_open(self, is_open: bool) -> None:
        """Sets to collapsed (False) or open (True). Requires a call to Window.SetNeedsLayout()
        afterwards, unless calling before window is visible."""
        ...

class ScrollableVert(Layout1D):
    """Scrollable vertical layout"""

    preferred_width: float
    """Sets the preferred width of the layout"""

    @overload
    def __init__(
        self,
        spacing: int = 0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets vertically, top to bottom,
        making their width equal to the layout's width. First argument is the
        spacing between widgets, the second is the margins. Both default to 0.
        """
        ...

    @overload
    def __init__(
        self,
        spacing: float = 0.0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that arranges widgets vertically, top to bottom,
        making their width equal to the layout's width. First argument is the
        spacing between widgets, the second is the margins. Both default to 0.
        """
        ...

class Checkbox(Widget):
    checked: bool
    
    def __init__(self, text: str) -> None:
        """Creates a checkbox with the given text"""
        ...

    def set_on_checked(self, callback: Callable[[bool], None]) -> None:
        """Calls passed function when checkbox changes state"""
        ...

class Label(Widget):
    """Displays text"""

    font_id: int
    """Set the font using the FontId returned from Application.add_font()"""
    text: str
    """The text of the label. Newlines will be treated as line breaks."""
    text_color: Color
    """The color of the text (gui.Color)"""

    def __init__(self, text: str) -> None:
        """Creates a Label with the given text"""
        ...

class CheckableTextTreeCell(Widget):
    """TreeView cell with a checkbox and text."""

    checkbox: Checkbox
    """Returns the checkbox widget (property is read-only)"""
    label: Label
    """Returns the label widget (property is read-only)"""


    def __init__(
        self,
        text: str,
        is_checked: bool,
        on_toggled: Callable[[bool], None],
    ) -> None:
        """Creates a TreeView cell with a checkbox and text.
        CheckableTextTreeCell(text, is_checked, on_toggled): on_toggled takes a boolean and returns None"""
        ...
    
class ColorEdit(Widget):
    """Color picker"""

    color_value: Color
    """Color value (gui.Color)"""

    def __init__(self) -> None: ...
    
    def set_on_value_changed(self, callback: Callable[[Color], None]) -> None:
        """Calls f(Color) when color changes by user input"""
        ...

class NumberEdit(Widget):
    """Allows the user to enter a number."""

    class Type(enum):
        """Enum class for NumberEdit types."""
        INT: int = 0
        DOUBLE: int = 1

    decimal_precision: int
    """Number of fractional digits shown"""
    double_value: float
    """Current value (double)"""
    int_value: int
    """Current value (int)"""
    maximum_value: float
    """The maximum value number can contain (read-only, use set_limits() to set)"""
    minimum_value: float
    """The minimum value number can contain (read-only, use set_limits() to set)"""

    def __init__(self, type: Type) -> None:
        """Creates a NumberEdit that is either integers (INT) or floating point (DOUBLE).
        The initial value is 0 and the limits are +/- max integer (roughly)."""
        ...

    def set_limits(self, min: float, max: float) -> None:
        """Sets the minimum and maximum values for the number"""
        ...
    
    def set_on_value_changed(self, callback: Callable[[float], None]) -> None:
        """Sets f(new_value) which is called with a Float when user changes widget's value"""
        ...
    
    @overload
    def set_preferred_width(self, width: int) -> None:
        """Sets the preferred width of the NumberEdit"""
        ...
    
    @overload
    def set_preferred_width(self, width: float) -> None:
        """Sets the preferred width of the NumberEdit"""
        ...
    
    def set_value(self, value: float) -> None:
        """Sets value"""
        ...

class TextEdit(Widget):
    """Allows the user to enter or modify text."""

    placeholder_text: str
    """The placeholder text displayed when text value is empty"""
    text_value: str
    """The value of text"""

    def __init__(self) -> None:
        """Creates a TextEdit widget with an initial value of an empty string."""
        ...

    def set_on_text_changed(self, callback: Callable[[str], None]) -> None:
        """Sets f(new_text) which is called whenever the the user makes a change to the text"""
        ...

    def set_on_value_changed(self, callback: Callable[[str], None]) -> None:
        """Sets f(new_text) which is called with the new text when the user completes text editing"""
        ...
    
class VectorEdit(Widget):
    """Allows the user to edit a 3-space vector"""

    vector_value: open3d.utility.Vector3dVector
    """Returns value [x, y, z]"""

    def __init__(self) -> None:
        ...

    def set_on_value_changed(self, callback: Callable[[open3d.utility.Vector3dVector], None]) -> None:
        """Sets f([x, y, z]) which is called whenever the user changes the value of a component"""
        ...

class ColormapTreeCell(Widget):
    """TreeView cell with a number edit and color edit"""

    color_edit: ColorEdit
    """Returns the ColorEdit widget (property is read-only)"""
    number_edit: NumberEdit
    """Returns the NumberEdit widget (property is read-only)"""

    def __init__(
        self,
        value: float,
        color: Color,
        on_value_changed: Callable[[float], None],
        on_color_changed: Callable[[Color], None],
    ) -> None:
        """Creates a TreeView cell with a number and a color edit.
        ColormapTreeCell(value, color, on_value_changed, on_color_changed):
        on_value_changed takes a double and returns None; on_color_changed takes a gui.Color and returns None"""
        ...

class LUTTreeCell(Widget):
    """TreeView cell with checkbox, text, and color edit"""

    checkbox: Checkbox
    """Returns the checkbox widget (property is read-only)"""
    color_edit: ColorEdit
    """Returns the ColorEdit widget (property is read-only)"""
    label: Label
    """Returns the label widget (property is read-only)"""

    def __init__(
        self,
        text: str,
        is_checked: bool,
        color: Color,
        on_enabled: Callable[[bool], None],
        on_color: Callable[[Color], None],
    ) -> None:
        """Creates a TreeView cell with a checkbox, text, and a color editor.
        LUTTreeCell(text, is_checked, color, on_enabled, on_color):
        on_enabled is called when the checkbox toggles, and takes a boolean and returns None;
        on_color is called when the user changes the color and it takes a gui.Color and returns None."""
        ...

class Button(Widget):

    horizontal_padding_em: float
    """Horizontal padding in em units"""
    is_on: bool
    """True if the button is toggleable and in the on state"""
    text: str
    """Gets/sets the button text."""
    toggleable: bool
    """True if button is toggleable, False if a push button"""
    vertical_padding_em: float
    """Vertical padding in em units"""

    def __init__(self, text: str) -> None:
        """Creates a button with the given text"""
        ...
    
    def set_on_clicked(self, callback: Callable[[], None]) -> None:
        """Calls passed function when button is pressed"""
        ...

class Combobox(Widget):
    """Exclusive selection from a pull-down menu"""

    number_of_items: int
    """The number of items (read-only)"""
    selected_index: int
    """The index of the currently selected item"""
    selected_text: str
    """The index of the currently selected item"""

    def __init__(self) -> None:
        """Creates an empty combobox. Use add_item() to add items"""
        ...
    
    def add_item(self, text: str) -> int:
        """Adds an item to the end"""
        ...

    @overload
    def change_item(self, index: int, new_text: str) -> None:
        """Changes the text of the item at index: change_item(index, new_text)"""
        ...
    
    @overload
    def change_item(self, text: str, new_text: str) -> None:
        """Changes the text of the matching item: change_item(text, new_text)"""
        ...
    
    def clear_items(self) -> None:
        """Removes all items"""
        ...
    
    def get_item(self, index: int) -> str:
        """Returns the item at the given index. Index must be valid."""
        ...
    
    @overload
    def remove_item(self, index: int) -> None:
        """Removes the item at the index"""
        ...

    @overload
    def remove_item(self, text: str) -> None:
        """Removes the first item of the given text"""
        ...
    
    def set_on_selection_changed(self, callback: Callable[[str, int], None]) -> None:
        """Calls f(str, int) when user selects item from combobox. Arguments are the selected text and selected index, respectively"""
        ...

class UIImage:
    """A bitmap suitable for displaying with ImageWidget"""

    class Scaling(enum):
        NONE = 0
        """No scaling"""
        ANY = 1
        """Scaled to fit"""
        ASPECT = 2
        """Scaled to fit but keeping the image's aspect ratio"""

    scaling: Scaling
    """Sets how the image is scaled:
    
    - `gui.UIImage.Scaling.NONE`: no scaling
    - `gui.UIImage.Scaling.ANY`: scaled to fit
    - `gui.UIImage.Scaling.ASPECT`: scaled to fit but keeping the image's aspect ratio"""

class ImageWidget(Widget):
    """Displays a bitmap"""

    ui_image: UIImage
    """Replaces the texture with a new texture. This is not a fast path,
    and is not recommended for video as you will exhaust internal texture resources."""

    @overload
    def __init__(self) -> None:
        """Creates an ImageWidget with no image"""
        ...
    
    @overload
    def __init__(self, path: str) -> None:
        """Creates an ImageWidget from the image at the specified path"""
        ...
    
    @overload
    def __init__(self, image: open3d.geometry.Image) -> None:
        """Creates an ImageWidget from the provided image"""
        ...
    
    @overload
    def __init__(self, image: open3d.t.geometry.Image) -> None:
        """Creates an ImageWidget from the provided t.geometry image"""
        ...
    
    def set_on_key(self, callback: Callable[[KeyEvent], int]) -> None:
        """Sets a callback for key events.

        This callback is passed a KeyEvent object. The callback must return
        EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, or
        EventCallackResult.CONSUMED.
        """
        ...
    
    def set_on_mouse(self, callback: Callable[[MouseEvent], int]) -> None:
        """Sets a callback for mouse events.

        This callback is passed a MouseEvent object. The callback must return
        EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, or
        EventCallackResult.CONSUMED.
        """
        ...
    
    @overload
    def update_image(self, image: open3d.geometry.Image) -> None:
        """Mostly a convenience function for ui_image.update_image().
        If 'image' is the same size as the current image, will update the
        texture with the contents of 'image'. This is the fastest path for
        setting an image, and is recommended if you are displaying video.
        If 'image' is a different size, it will allocate a new texture,
        which is essentially the same as creating a new UIImage and calling
        SetUIImage(). This is the slow path, and may eventually exhaust
        internal texture resources.
        """
        ...
    
    @overload
    def update_image(self, image: open3d.t.geometry.Image) -> None:
        """Mostly a convenience function for ui_image.update_image().
        If 'image' is the same size as the current image, will update the
        texture with the contents of 'image'. This is the fastest path for
        setting an image, and is recommended if you are displaying video.
        If 'image' is a different size, it will allocate a new texture,
        which is essentially the same as creating a new UIImage and calling
        SetUIImage(). This is the slow path, and may eventually exhaust
        internal texture resources.
        """
        ...

class Label3D:
    """Displays text in a 3D scene"""

    color: Color
    """The color of the text (gui.Color)"""
    position: np.ndarray[np.float32[3, 1]]
    """The position of the text in 3D coordinates.
    Type: numpy.ndarray[numpy.float32[3, 1]]"""
    scale: float
    """The scale of the 3D label. When set to 1.0 (the default) text
    will be rendered at its native font size. Larger and smaller values
    of scale will enlarge or shrink the rendered text. Note"""
    text: str
    """The text to display with this label."""

    def __init__(self, text: str, position: np.ndarray[np.float32[3, 1]]) -> None:
        """Create a 3D Label with given text and position.

        Args:
            text (str): The text to display with this label.
            position (np.ndarray[np.float32[3, 1]]): The position of the text in 3D coordinates.
        """
        ...

class ListView(Widget):
    """Displays a list of text"""

    selected_index: int
    """The index of the currently selected item"""
    selected_value: str
    """The text of the currently selected item"""

    def __init__(self) -> None:
        """Creates an empty list"""
        ...

    def set_items(self, items: list[str]) -> None:
        """Sets the list to display the list of items provided"""
        ...

    def set_max_visible_items(self, num: int) -> None:
        """Limit the max visible items shown to user. Set to negative number
        will make list extends vertically as much as possible, otherwise the
        list will at least show 3 items and at most show num items."""
        ...

    def set_on_selection_changed(self, callback: Callable[[str, bool], None]) -> None:
        """Calls f(new_val, is_double_click) when user changes selection"""
        ...

class TreeView(Widget):
    """Hierarchical list"""

    can_select_items_with_children: bool
    """If set to False, clicking anywhere on an item with will toggle the item
    open or closed; the item cannot be selected. If set to True, items with
    children can be selected, and to toggle open/closed requires clicking the
    arrow or double-clicking the item"""
    selected_item: int
    """The currently selected item"""

    def __init__(self) -> None:
        """Creates an empty TreeView widget"""
        ...
    
    def add_item(self, parent: int, widget: Widget) -> int:
        """Adds a child item to the parent. add_item(parent, widget)"""
        ...
    
    def add_text_item(self, parent: int, text: str) -> int:
        """Adds a child item to the parent. add_text_item(parent, text)"""
        ...
    
    def clear(self) -> None:
        """Removes all items"""
        ...

    def get_item(self, item_id: int) -> Widget:
        """Returns the widget associated with the provided Item ID. For example,
        to manipulate the widget of the currently selected item you would use
        the ItemID of the selected_item property with get_item to get the widget.
        """
        ...

    def get_root_item(self) -> int:
        """Returns the root item. This item is invisible, so its child are the top-level items"""
        ...
    
    def remove_item(self, item_id: int) -> None:
        """Removes an item and all its children (if any)"""
        ...

    def set_on_selection_changed(self, callback: Callable[[int], None]) -> None:
        """Sets f(new_item_id) which is called when the user changes the selection."""
        ...

class VGrid(Widget):
    """Grid layout"""

    margins: Margins
    """Returns the margins"""
    preferred_width: float
    """Sets the preferred width of the layout"""
    spacing: float
    """Returns the spacing between rows and columns"""

    @overload
    def __init__(
        self,
        cols: int,
        spacing: int = 0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that orders its children in a grid, left to right, top to bottom,
        according to the number of columns. The first argument is the number of columns, the
        second is the spacing between items (both vertically and horizontally), and third is
        the margins. Both spacing and margins default to zero."""
        ...

    @overload
    def __init__(
        self,
        cols: int,
        spacing: float = 0.0,
        margins: Margins = Margins(),
    ) -> None:
        """Creates a layout that orders its children in a grid, left to right, top to bottom,
        according to the number of columns. The first argument is the number of columns, the
        second is the spacing between items (both vertically and horizontally), and third is
        the margins. Both spacing and margins default to zero."""
        ...

class TabControl(Widget):
    """Tab control"""

    selected_tab_index: int
    """The index of the currently selected item"""

    def __init__(self) -> None:
        """Creates an empty tab control"""
        ...

    def add_tab(self, title: str, widget: Widget) -> None:
        """Adds a tab. The first parameter is the title of the tab, and the second parameter
        is a widget-normally this is a layout."""
        ...

    def set_on_selected_tab_changed(self, callback: Callable[[int], None]) -> None:
        """Calls the provided callback function with the index of the currently selected tab
        whenever the user clicks on a different tab"""
        ...

class StackedWidget(Widget):
    """Like a TabControl but without the tabs"""

    selected_index: int
    """Selects the index of the child to display"""

    def __init__(self) -> None:
        """Creates an empty stacked widget"""
        ...

class ProgressBar(Widget):
    """Displays a progress bar"""

    value: float
    """The value of the progress bar, ranges from 0.0 to 1.0"""

    def __init__(self) -> None: ...
    
class RadioButton(Widget):
    """Exclusive selection from radio button list"""

    class Type(enum):
        """Enum class for RadioButton types."""
        VERT: int = 0
        HORIZ: int = 1

    selected_index: int
    """The index of the currently selected item"""
    selected_value: str
    """The text of the currently selected item"""

    def __init__(self, type: Type) -> None:
        """Creates an empty radio buttons. Use set_items() to add items"""
        ...

    def set_items(self, items: list[str]) -> None:
        """Set radio items, each item is a radio button."""
        ...

    def set_on_selection_changed(self, callback: Callable[[int], None]) -> None:
        """Calls f(new_idx) when user changes selection"""
        ...

class Slider(Widget):
    """A slider widget for visually selecting numbers."""

    class Type(enum):
        """Enum class for Slider types."""
        INT: int = 0
        DOUBLE: int = 1

    double_value: float
    """Slider value (double)"""
    int_value: int
    """Slider value (int)"""
    get_maximum_value: float
    """The maximum value number can contain (read-only, use set_limits() to set)"""
    get_minimum_value: float
    """The minimum value number can contain (read-only, use set_limits() to set)"""

    def __init__(self, type: Type) -> None:
        """Creates a NumberEdit that is either integers (INT) or floating point (DOUBLE).
        The initial value is 0 and the limits are +/- infinity."""
        ...
    
    def set_limits(self, min: float, max: float) -> None:
        """Sets the minimum and maximum values for the slider"""
        ...

    def set_on_value_changed(self, callback: Callable[[float], None]) -> None:
        """Sets f(new_value) which is called with a Float when user changes widget's value"""
        ...

class ToggleSwitch(Widget):

    is_on: bool
    """True if is one, False otherwise"""

    def __init__(self, text: str) -> None:
        """Creates a toggle switch with the given text"""
        ...

    def set_on_clicked(self, callback: Callable[[bool], None]) -> None:
        """Sets f(is_on) which is called when the switch changes state."""
        ...

class WidgetProxy(Widget):
    """Widget container to delegate any widget dynamically. Widget can not be managed dynamically.
    Although it is allowed to add more child widgets, it's impossible to replace some child with
    new on or remove children. WidgetProxy is designed to solve this problem. When WidgetProxy is
    created, it's invisible and disabled, so it won't be drawn or layout, seeming like it does not
    exist. When a widget is set by set_widget, all Widget's APIs will be conducted to that child widget.
    It looks like WidgetProxy is that widget. At any time, a new widget could be set, to replace the old one.
    and the old widget will be destroyed. Due to the content changing after a new widget is set or cleared,
    a re-layout of Window might be called after set_widget. The delegated widget could be retrieved by get_widget
    in case you need to access it directly, like get check status of a CheckBox. API other than set_widget
    and get_widget has completely same functions as Widget."""

    def __init__(self) -> None:
        """Creates a widget proxy"""
        ...
    
    def get_widget(self) -> Widget:
        """Retrieve current delegated widget.return instance of current delegated widget set by set_widget.
        An empty pointer will be returned if there is none."""
        ...

    def set_widget(self, widget: Widget) -> None:
        """set a new widget to be delegated by this one. After set_widget, the previously delegated widget ,
        will be abandon all calls to Widget's API will be conducted to widget. Before any set_widget call,
        this widget is invisible and disabled, seems it does not exist because it won't be drawn or in a layout."""
        ...

class WidgetStack(Widget):
    """A widget stack saves all widgets pushed into by push_widget and always shows the top one.
    The WidgetStack is a subclass of WidgetProxy, in otherwords, the topmost widget will delegate
    itself to WidgetStack. pop_widget will remove the topmost widget and callback set by set_on_top
    taking the new topmost widget will be called. The WidgetStack disappears in GUI if there is no
    widget in stack."""

    def __init__(self) -> None:
        """Creates a widget stack. The widget stack without anywidget will not be shown in GUI
        until set_widget iscalled to push a widget."""
        ...
    
    def pop_widget(self) -> Widget:
        """pop the topmost widget in the stack. The new topmost widgetof stack will be the
        widget on the show in GUI."""
        ...

    def push_widget(self, widget: Widget) -> None:
        """push a new widget onto the WidgetStack's stack, hiding whatever widget was there
        before and making the new widget visible."""
        ...

    def set_on_top(self, callback: Callable[[Widget], None]) -> None:
        """Callable[[widget] -> None], called while a widget becomes the topmost of stack after
        some widget is poppedout. It won't be called if a widget is pushed into stack by set_widget."""
        ...

    def set_widget(self, widget: Widget) -> None:
        """set a new widget to be delegated by this one. After set_widget, the previously delegated
        widget , will be abandon all calls to Widget's API will be conducted to widget. Before any
        set_widget call, this widget is invisible and disabled, seems it does not exist because it
        won't be drawn or in a layout."""
        ...

class SceneWidget(Widget):
    """Displays 3D content"""

    class Controls(enum):
        """Enum class describing mouse interaction."""
        ROTATE_CAMERA: int = 0
        """Rotate the camera around the center of rotation"""
        ROTATE_CAMERA_SPHERE: int = 1
        """Rotate the camera around the center of rotation, but constrained to a sphere around the center of rotation"""
        FLY: int = 2
        """Fly around the scene"""
        ROTATE_SUN: int = 3
        """Rotate the sun around the center of rotation"""
        ROTATE_IBL: int = 4
        """Rotate the IBL around the center of rotation"""
        PICK_POINTS: int = 6
        """Pick points"""

    center_of_rotation: np.ndarray[np.float32[3, 1]]
    """Current center of rotation (for ROTATE_CAMERA and ROTATE_CAMERA_SPHERE)
    Type: numpy.ndarray[numpy.float32[3, 1]]"""
    scene: open3d.visualization.rendering.Scene
    """The rendering.Open3DScene that the SceneWidget renders"""

    def __init__(self) -> None:
        """Creates an empty SceneWidget. Assign a Scene with the 'scene' property"""
        ...

    def add_3d_label(self, position: np.ndarray[np.float32[3, 1]], text: str) -> Label3D:
        """Add a 3D text label to the scene. The label will be anchored at the specified 3D point."""
        ...

    def enable_scene_caching(self, enable: bool) -> None:
        """Enable/Disable caching of scene content when the view or model is not changing.
        Scene caching can help improve UI responsiveness for large models and point clouds"""
        ...
    
    def force_redraw(self) -> None:
        """Ensures scene redraws even when scene caching is enabled."""
        ...
    
    def look_at(
            self,
            center: np.ndarray[np.float32[3, 1]],
            eye: np.ndarray[np.float32[3, 1]],
            up: np.ndarray[np.float32[3, 1]]
    ) -> None:
        """look_at(center, eye, up): sets the camera view so that the camera is located at 'eye',
        pointing towards 'center', and oriented so that the up vector is 'up'"""
        ...

    def remove_3d_label(self, label: Label3D) -> None:
        """Removes the 3D text label from the scene"""
        ...

    def set_on_key(self, callback: Callable[[KeyEvent], int]) -> None:
        """Sets a callback for key events.

        This callback is passed a KeyEvent object. The callback must return
        EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, or
        EventCallbackResult.CONSUMED.
        """
        ...

    def set_on_mouse(self, callback: Callable[[MouseEvent], int]) -> None:
        """Sets a callback for mouse events.

        This callback is passed a MouseEvent object. The callback must return
        EventCallbackResult.IGNORED, EventCallbackResult.HANDLED, or
        EventCallbackResult.CONSUMED.
        """
        ...

    def set_on_sun_direction_changed(self, callback: Callable[[np.ndarray[np.float32[3, 1]]], None]) -> None:
        """Callback when user changes sun direction (only called in ROTATE_SUN control mode).
        Called with one argument, the [i, j, k] vector of the new sun direction

        Args:
            callback (Callable[[np.ndarray[np.float32[3, 1]]], None]): [description]
        """
        ...

    def set_view_controls(self, controls: Controls) -> None:
        """Sets mouse interaction, e.g. ROTATE_OBJ"""
        ...

    @overload
    def setup_camera(
        self,
        field_of_view: float,
        model_bounds: open3d.geometry.AxisAlignedBoundingBox,
        center_of_rotation: np.ndarray[np.float32[3, 1]],
    ) -> None:
        """Configure the camera: setup_camera(field_of_view, model_bounds, center_of_rotation).

        Args:
            field_of_view (float): The field of view
            model_bounds (open3d.geometry.AxisAlignedBoundingBox): The bounding box of the model
            center_of_rotation (np.ndarray[np.float32[3, 1]]): The center of rotation
        """
        ...
    
    @overload
    def setup_camera(
        self,
        intrinsics: open3d.camera.PinholeCameraIntrinsic,
        extrinsic_matrix: np.ndarray[np.float64[4, 4]],
        model_bounds: open3d.geometry.AxisAlignedBoundingBox,
    ) -> None:
        """setup_camera(intrinsics, extrinsic_matrix, model_bounds): sets the camera view

        Args:
            intrinsics (open3d.camera.PinholeCameraIntrinsic): The camera intrinsics
            extrinsic_matrix (np.ndarray[np.float64[4, 4]]): The camera extrinsics
            model_bounds (open3d.geometry.AxisAlignedBoundingBox): The bounding box of the model
        """
        ...

    @overload
    def setup_camera(
        self,
        intrinsic_matrix: np.ndarray[np.float64[3, 3]],
        extrinsic_matrix: np.ndarray[np.float64[4, 4]],
        intrinsic_width_px: int,
        intrinsic_height_px: int,
        model_bounds: open3d.geometry.AxisAlignedBoundingBox,
    ) -> None:
        """setup_camera(intrinsic_matrix, extrinsic_matrix, intrinsic_width_px, intrinsic_height_px, model_bounds):
        sets the camera view

        Args:
            intrinsic_matrix (np.ndarray[np.float64[3, 3]]): The camera intrinsics
            extrinsic_matrix (np.ndarray[np.float64[4, 4]]): The camera extrinsics
            intrinsic_width_px (int): The width of the camera intrinsics
            intrinsic_height_px (int): The height of the camera intrinsics
            model_bounds (open3d.geometry.AxisAlignedBoundingBox): The bounding box of the model
        """
        ...
