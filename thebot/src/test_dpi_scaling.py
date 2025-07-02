"""
Test script to verify DPI/scaling and region math for debugging.
"""
from capture import get_actual_display_resolution, get_dpi_scale

def test_display_info():
    w, h = get_actual_display_resolution()
    scale_x, scale_y = get_dpi_scale()
    print(f"Actual display resolution: {w}x{h}")
    print(f"DPI scaling factor: x={scale_x:.2f} ({scale_x*100:.0f}%), y={scale_y:.2f} ({scale_y*100:.0f}%)")
    # Test region math
    fov = 280
    left = (w - fov) // 2
    top = (h - fov) // 2
    print(f"Centered region: left={left}, top={top}, width={fov}, height={fov}")
    # Clamp test
    left = max(0, min(left, w - fov))
    top = max(0, min(top, h - fov))
    print(f"Clamped region: left={left}, top={top}, width={fov}, height={fov}")

if __name__ == "__main__":
    test_display_info()
