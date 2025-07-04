"""
Production-Grade Overlay System with Hardware Acceleration and Transparency
Purpose: High-performance in-game overlay with minimal performance impact
"""
import pygame
import pygame.gfxdraw
import numpy as np
import threading
import time
import queue
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import platform
import ctypes
from pathlib import Path

try:
    import OpenGL.GL as gl
    from pygame.locals import OPENGL, DOUBLEBUF
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    # Define fallback constants if OpenGL is not available
    OPENGL = 0
    DOUBLEBUF = 0

# Handle both direct execution and module import
try:
    from .logger_util import get_logger
    from .config_manager import ConfigManager, ScreenRegion
    from .detect import Detection
except ImportError:
    from logger_util import get_logger
    from config_manager import ConfigManager, ScreenRegion
    from detect import Detection


class OverlayMode(Enum):
    """Overlay rendering modes"""
    WINDOWED = auto()      # Normal window
    BORDERLESS = auto()    # Borderless window
    TRANSPARENT = auto()   # Transparent overlay
    OPENGL = auto()        # Hardware accelerated


class RenderLayer(Enum):
    """Rendering layers for organization"""
    BACKGROUND = 0
    DETECTION = 1
    CROSSHAIR = 2
    HUD = 3
    MENU = 4
    DEBUG = 5


@dataclass
class OverlayTheme:
    """Overlay visual theme"""
    # Colors (RGBA)
    primary: Tuple[int, int, int, int] = (0, 255, 0, 255)          # Green
    secondary: Tuple[int, int, int, int] = (255, 255, 0, 255)      # Yellow
    danger: Tuple[int, int, int, int] = (255, 0, 0, 255)           # Red
    info: Tuple[int, int, int, int] = (0, 255, 255, 255)           # Cyan
    background: Tuple[int, int, int, int] = (0, 0, 0, 128)         # Semi-transparent black
    text: Tuple[int, int, int, int] = (255, 255, 255, 255)         # White
    
    # Sizes
    line_width: int = 2
    font_size: int = 14
    corner_radius: int = 5
    
    # Effects
    enable_glow: bool = True
    glow_radius: int = 3
    enable_animations: bool = True
    animation_speed: float = 1.0


@dataclass
class RenderElement:
    """Base render element"""
    layer: RenderLayer
    position: Tuple[int, int]
    visible: bool = True
    opacity: float = 1.0
    
    def render(self, surface: pygame.Surface, theme: OverlayTheme) -> None:
        """Render element to surface"""
        raise NotImplementedError


@dataclass
class DetectionBox(RenderElement):
    """Detection bounding box render element"""
    detection: Optional[Detection] = field(default=None)
    color_override: Optional[Tuple[int, int, int, int]] = field(default=None)
    show_label: bool = field(default=True)
    
    def __post_init__(self):
        # Ensure detection is provided
        if self.detection is None:
            raise ValueError("detection field is required")
    
    def render(self, surface: pygame.Surface, theme: OverlayTheme) -> None:
        """Render detection box"""
        if not self.visible or self.opacity <= 0 or self.detection is None:
            return
        
        x1, y1, x2, y2 = map(int, self.detection.box)
        color = self.color_override or theme.primary
        
        # Apply opacity
        color = (*color[:3], int(color[3] * self.opacity))
        
        # Draw box with corners
        if theme.enable_glow:
            # Draw glow effect
            for i in range(theme.glow_radius):
                alpha = int(50 * (1 - i / theme.glow_radius) * self.opacity)
                glow_color = (*color[:3], alpha)
                pygame.draw.rect(surface, glow_color, 
                               (x1-i, y1-i, x2-x1+2*i, y2-y1+2*i), 
                               theme.line_width, border_radius=theme.corner_radius)
        
        # Draw main box
        pygame.draw.rect(surface, color, (x1, y1, x2-x1, y2-y1), 
                        theme.line_width, border_radius=theme.corner_radius)
        
        # Draw corner accents
        corner_length = 20
        corners = [
            # Top-left
            [(x1, y1 + corner_length), (x1, y1), (x1 + corner_length, y1)],
            # Top-right
            [(x2 - corner_length, y1), (x2, y1), (x2, y1 + corner_length)],
            # Bottom-left
            [(x1, y2 - corner_length), (x1, y2), (x1 + corner_length, y2)],
            # Bottom-right
            [(x2 - corner_length, y2), (x2, y2), (x2, y2 - corner_length)]
        ]
        
        for corner in corners:
            pygame.draw.lines(surface, color, False, corner, theme.line_width + 1)
        
        # Draw label
        if self.show_label and self.detection and hasattr(self.detection, 'class_name') and hasattr(self.detection, 'confidence'):
            font = pygame.font.Font(None, theme.font_size)
            label = f"{self.detection.class_name} {self.detection.confidence:.2f}"
            
            # Render text with background
            text_surface = font.render(label, True, theme.text)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (x1, y1 - theme.font_size - 4)
            
            # Draw background
            bg_rect = text_rect.inflate(8, 4)
            pygame.draw.rect(surface, (*theme.background[:3], 200), bg_rect, border_radius=3)
            
            # Draw text
            surface.blit(text_surface, text_rect)


@dataclass
class Crosshair(RenderElement):
    """Crosshair render element"""
    style: str = "dot"  # 'dot', 'cross', 'circle', 'custom'
    size: int = 10
    thickness: int = 2
    gap: int = 5
    dot_size: int = 2
    
    def render(self, surface: pygame.Surface, theme: OverlayTheme) -> None:
        """Render crosshair"""
        if not self.visible or self.opacity <= 0:
            return
        
        cx, cy = self.position
        color = (*theme.primary[:3], int(255 * self.opacity))
        
        if self.style == "dot":
            # Simple dot
            pygame.gfxdraw.filled_circle(surface, cx, cy, self.dot_size, color)
            
        elif self.style == "cross":
            # Classic cross
            # Horizontal line
            pygame.draw.line(surface, color,
                           (cx - self.size - self.gap, cy),
                           (cx - self.gap, cy), self.thickness)
            pygame.draw.line(surface, color,
                           (cx + self.gap, cy),
                           (cx + self.size + self.gap, cy), self.thickness)
            
            # Vertical line
            pygame.draw.line(surface, color,
                           (cx, cy - self.size - self.gap),
                           (cx, cy - self.gap), self.thickness)
            pygame.draw.line(surface, color,
                           (cx, cy + self.gap),
                           (cx, cy + self.size + self.gap), self.thickness)
            
            # Center dot
            if self.dot_size > 0:
                pygame.gfxdraw.filled_circle(surface, cx, cy, self.dot_size, color)
                
        elif self.style == "circle":
            # Circle crosshair
            pygame.gfxdraw.circle(surface, cx, cy, self.size, color)
            if self.thickness > 1:
                for i in range(1, self.thickness):
                    pygame.gfxdraw.circle(surface, cx, cy, self.size + i, color)
            
            # Center dot
            if self.dot_size > 0:
                pygame.gfxdraw.filled_circle(surface, cx, cy, self.dot_size, color)


@dataclass
class HUDElement(RenderElement):
    """HUD information display"""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, surface: pygame.Surface, theme: OverlayTheme) -> None:
        """Render HUD"""
        if not self.visible or self.opacity <= 0:
            return
        
        font = pygame.font.Font(None, theme.font_size)
        x, y = self.position
        line_height = theme.font_size + 4
        
        # Background
        lines = [
            f"FPS: {self.metrics.get('fps', 0):.1f}",
            f"Latency: {self.metrics.get('latency', 0):.1f}ms",
            f"Targets: {self.metrics.get('targets', 0)}",
            f"State: {self.metrics.get('state', 'Ready')}"
        ]
        
        # Calculate background size
        max_width = max(font.size(line)[0] for line in lines) + 16
        bg_height = len(lines) * line_height + 8
        
        # Draw background
        bg_surface = pygame.Surface((max_width, bg_height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, (*theme.background[:3], int(200 * self.opacity)),
                        (0, 0, max_width, bg_height), border_radius=theme.corner_radius)
        surface.blit(bg_surface, (x, y))
        
        # Draw text
        for i, line in enumerate(lines):
            text_color = (*theme.text[:3], int(255 * self.opacity))
            text_surface = font.render(line, True, text_color)
            surface.blit(text_surface, (x + 8, y + 4 + i * line_height))


class TransparencyManager:
    """Platform-specific window transparency"""
    
    def __init__(self, window_handle: int):
        self.handle = window_handle
        self.platform = platform.system()
        self.logger = get_logger(__name__)
    
    def set_transparent(self) -> bool:
        """Make window transparent"""
        try:
            if self.platform == "Windows":
                return self._set_windows_transparent()
            elif self.platform == "Linux":
                return self._set_linux_transparent()
            else:
                self.logger.warning(f"Transparency not supported on {self.platform}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to set transparency: {e}")
            return False
    
    def _set_windows_transparent(self) -> bool:
        """Set Windows window transparency"""
        try:
            # Windows constants
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            LWA_COLORKEY = 0x00000001
            LWA_ALPHA = 0x00000002
            
            # Get window handle
            hwnd = self.handle
            
            # Set window style
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            
            # Set transparency
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd, 0x000000, 0, LWA_COLORKEY
            )
            
            # Set topmost
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Windows transparency error: {e}")
            return False
    
    def _set_linux_transparent(self) -> bool:
        """Set Linux window transparency (X11)"""
        # Would implement X11 transparency here
        self.logger.warning("Linux transparency not implemented")
        return False


class OverlayRenderer:
    """High-performance overlay renderer"""
    
    def __init__(self, 
                 width: int, 
                 height: int,
                 mode: OverlayMode,
                 theme: OverlayTheme):
        
        self.width = width
        self.height = height
        self.mode = mode
        self.theme = theme
        self.logger = get_logger(__name__)
        
        # Render elements organized by layer
        self.elements: Dict[RenderLayer, List[RenderElement]] = {
            layer: [] for layer in RenderLayer
        }
        
        # Performance tracking
        self.frame_times = []
        self.last_fps_calc = time.time()
        self.current_fps = 0.0
        
        # Initialize renderer
        self._init_renderer()
    
    def _init_renderer(self) -> None:
        """Initialize rendering backend"""
        if self.mode == OverlayMode.OPENGL and OPENGL_AVAILABLE:
            self._init_opengl()
        else:
            self._init_software()
    
    def _init_software(self) -> None:
        """Initialize software renderer"""
        # Create render surface
        self.render_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
    def _init_opengl(self) -> None:
        """Initialize OpenGL renderer"""
        # Set OpenGL attributes
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        
        # Initialize OpenGL
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0, 0, 0, 0)
    
    def add_element(self, element: RenderElement) -> None:
        """Add render element"""
        self.elements[element.layer].append(element)
    
    def remove_element(self, element: RenderElement) -> None:
        """Remove render element"""
        if element.layer in self.elements:
            try:
                self.elements[element.layer].remove(element)
            except ValueError:
                pass
    
    def clear_layer(self, layer: RenderLayer) -> None:
        """Clear all elements in layer"""
        self.elements[layer].clear()
    
    def render(self, screen: pygame.Surface) -> None:
        """Render all elements"""
        start_time = time.perf_counter()
        
        # Clear
        if self.mode == OverlayMode.TRANSPARENT:
            screen.fill((0, 0, 0, 0))
        else:
            screen.fill((0, 0, 0))
        
        # Render layers in order
        for layer in RenderLayer:
            for element in self.elements[layer]:
                if element.visible:
                    element.render(screen, self.theme)
        
        # Track performance
        frame_time = (time.perf_counter() - start_time) * 1000
        self.frame_times.append(frame_time)
        
        # Calculate FPS
        now = time.time()
        if now - self.last_fps_calc >= 1.0:
            if self.frame_times:
                self.current_fps = 1000.0 / (sum(self.frame_times) / len(self.frame_times))
                self.frame_times.clear()
            self.last_fps_calc = now
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps


class OverlaySystem:
    """
    Production overlay system with multi-threaded rendering
    """
    
    def __init__(self,
                 mode: OverlayMode = OverlayMode.TRANSPARENT,
                 config_manager: Optional[ConfigManager] = None,
                 theme: Optional[OverlayTheme] = None):
        
        self.logger = get_logger(__name__)
        self.mode = mode
        self.config_manager = config_manager or ConfigManager()
        self.theme = theme or OverlayTheme()
        
        # Get screen region
        self.region = self.config_manager.get_screen_region()
        
        # Pygame setup
        pygame.init()
        self.screen: Optional[pygame.Surface] = None
        self.clock = pygame.time.Clock()
        
        # Renderer
        self.renderer: Optional[OverlayRenderer] = None
        
        # State
        self.running = False
        self.visible = True
        self.transparency_enabled = False
        
        # Data queues
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.detection_queue: queue.Queue[List[Detection]] = queue.Queue(maxsize=10)
        self.metrics_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=10)
        
        # Threading
        self.render_thread: Optional[threading.Thread] = None
        
        # Initialize
        self._init_display()
        self._init_renderer()
        self._setup_default_elements()
    
    def _init_display(self) -> None:
        """Initialize display"""
        flags = 0
        
        if self.mode == OverlayMode.TRANSPARENT:
            flags |= pygame.NOFRAME
        elif self.mode == OverlayMode.BORDERLESS:
            flags |= pygame.NOFRAME
        elif self.mode == OverlayMode.OPENGL and OPENGL_AVAILABLE:
            flags |= OPENGL | DOUBLEBUF
        
        # Create window
        self.screen = pygame.display.set_mode(
            (self.region.width, self.region.height),
            flags
        )
        
        pygame.display.set_caption("AI Aim Assist Overlay")
        
        # Set window position
        if platform.system() == "Windows":
            hwnd = pygame.display.get_wm_info()["window"]
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0,
                self.region.left, self.region.top,
                0, 0, 0x0001
            )
            
            # Enable transparency if requested
            if self.mode == OverlayMode.TRANSPARENT:
                transparency_mgr = TransparencyManager(hwnd)
                self.transparency_enabled = transparency_mgr.set_transparent()
    
    def _init_renderer(self) -> None:
        """Initialize renderer"""
        self.renderer = OverlayRenderer(
            self.region.width,
            self.region.height,
            self.mode,
            self.theme
        )
    
    def _setup_default_elements(self) -> None:
        """Setup default overlay elements"""
        if self.renderer is None:
            return
            
        # Crosshair
        crosshair = Crosshair(
            layer=RenderLayer.CROSSHAIR,
            position=(self.region.width // 2, self.region.height // 2),
            style="cross",
            size=15,
            thickness=2,
            gap=5,
            dot_size=2
        )
        self.renderer.add_element(crosshair)
        
        # HUD
        hud = HUDElement(
            layer=RenderLayer.HUD,
            position=(10, 10)
        )
        self.renderer.add_element(hud)
        self.hud_element = hud  # Keep reference for updates
    
    def run(self) -> None:
        """Run overlay in current thread"""
        self.running = True
        
        while self.running:
            try:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F10:
                            self.toggle_visibility()
                
                if not self.visible:
                    self.clock.tick(10)  # Low FPS when hidden
                    continue
                
                # Update data
                self._update_detections()
                self._update_metrics()
                
                # Render
                if self.screen is not None and self.renderer is not None:
                    self.renderer.render(self.screen)
                
                # Update display
                pygame.display.flip()
                self.clock.tick(144)  # Cap at 144 FPS
                
            except Exception as e:
                self.logger.error(f"Overlay render error: {e}")
        
        self._cleanup()
    
    def start_threaded(self) -> None:
        """Start overlay in separate thread"""
        self.render_thread = threading.Thread(
            target=self.run,
            name="OverlayRenderer",
            daemon=True
        )
        self.render_thread.start()
    
    def stop(self) -> None:
        """Stop overlay"""
        self.running = False
        
        if self.render_thread:
            self.render_thread.join(timeout=2.0)
    
    def _update_detections(self) -> None:
        """Update detection displays"""
        try:
            detections = self.detection_queue.get_nowait()
            
            # Clear old detections
            if self.renderer is not None:
                self.renderer.clear_layer(RenderLayer.DETECTION)
            
                # Add new detections
                for detection in detections:
                    box = DetectionBox(
                        layer=RenderLayer.DETECTION,
                        position=(0, 0),
                        detection=detection
                    )
                    self.renderer.add_element(box)
                
        except queue.Empty:
            pass
    
    def _update_metrics(self) -> None:
        """Update HUD metrics"""
        try:
            metrics = self.metrics_queue.get_nowait()
            
            # Add overlay FPS
            if self.renderer is not None:
                metrics['fps'] = self.renderer.get_fps()
            
            # Update HUD
            if hasattr(self, 'hud_element'):
                self.hud_element.metrics = metrics
                
        except queue.Empty:
            pass
    
    def update_frame(self, frame: np.ndarray) -> None:
        """Update frame data (for future use)"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def update_detections(self, detections: List[Detection]) -> None:
        """Update detection data"""
        try:
            self.detection_queue.put_nowait(detections)
        except queue.Full:
            # Drop oldest
            try:
                self.detection_queue.get_nowait()
                self.detection_queue.put_nowait(detections)
            except queue.Empty:
                pass
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics data"""
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            # Drop oldest
            try:
                self.metrics_queue.get_nowait()
                self.metrics_queue.put_nowait(metrics)
            except queue.Empty:
                pass
    
    def toggle_visibility(self) -> None:
        """Toggle overlay visibility"""
        self.visible = not self.visible
        
        if platform.system() == "Windows" and self.screen:
            hwnd = pygame.display.get_wm_info()["window"]
            
            if self.visible:
                ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
            else:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    
    def set_theme(self, theme: OverlayTheme) -> None:
        """Update overlay theme"""
        self.theme = theme
        if self.renderer:
            self.renderer.theme = theme
    
    def _cleanup(self) -> None:
        """Cleanup resources"""
        pygame.quit()
        self.logger.info("Overlay system cleaned up")