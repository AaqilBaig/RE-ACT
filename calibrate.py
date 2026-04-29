import json
import time
import os
from obj_det import locate_and_segment

def get_point(prompt_msg, physical_x, physical_y, target_class="coin"):
    print(f"\n{prompt_msg}")
    print(f"Target physical coordinates: X={physical_x}mm, Y={physical_y}mm")
    input("Press Enter when the object is in position...")
    
    coords = None
    while not coords:
        coords = locate_and_segment(target_class)
        if not coords:
            retry = input("Failed to detect object. Type 'r' to retry or 'q' to quit: ").strip().lower()
            if retry == 'q':
                exit()
    print(f"Recorded pixel coordinates: {coords} for physical ({physical_x}, {physical_y})")
    return coords

def main():
    print("=== Robotic Arm Vision Calibration ===")
    print("This script will guide you to place an object at 3 specific coordinates.")
    print("This will calculate the exact scale and camera offset automatically.")
    
    target_class = input("What object will you use for calibration? (default 'coin'): ").strip()
    if not target_class:
        target_class = "coin"
        
    print("\nMeasurements represent real physical millimeters from the base of the robotic arm (X=0, Y=0).")
    
    # Point 1: Center line, forward
    # 150mm is a safe distance considering your 86.2mm workspace offset from base
    p1_phys_x, p1_phys_y = 0.0, 150.0  
    p1_px = get_point(f"1. Place the {target_class} exactly in the center of the X-axis (X=0), and 150mm forward from the arm's base (Y=150).", p1_phys_x, p1_phys_y, target_class)
    
    # Point 2: Right side
    p2_phys_x, p2_phys_y = 100.0, 150.0
    p2_px = get_point(f"2. Move the {target_class} exactly 100mm to the RIGHT (X=100, Y=150).", p2_phys_x, p2_phys_y, target_class)
    
    # Point 3: Center line, further forward
    p3_phys_x, p3_phys_y = 0.0, 250.0
    p3_px = get_point(f"3. Move the {target_class} back to the center line, but 250mm forward from the arm's base (X=0, Y=250).", p3_phys_x, p3_phys_y, target_class)
    
    # Calculate scaling and offsets
    # Based on your setup, Camera X = Physical X, and Camera Y = Physical Y.
    # Also, moving Physical Right (+X) moves Camera Left (-X), which will result in a negative scale_x.
    
    # Scale X: moving right (physical X) changes pixel X
    dx_px = p2_px[0] - p1_px[0]
    if dx_px == 0:
        print("Error: X pixel did not change. Calibration failed. Ensure the object was moved right.")
        return
    scale_x = (p2_phys_x - p1_phys_x) / dx_px
    
    # Scale Y: moving forward (physical Y) changes pixel Y
    dy_px = p3_px[1] - p1_px[1]
    if dy_px == 0:
        print("Error: Y pixel did not change. Calibration failed. Ensure the object was moved forward.")
        return
    scale_y = (p3_phys_y - p1_phys_y) / dy_px
    
    # Find base (0,0) in pixels using origin (Point 1)
    base_px_x = p1_px[0] - (p1_phys_x / scale_x)
    base_px_y = p1_px[1] - (p1_phys_y / scale_y)
    
    cal_data = {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "base_px_x": base_px_x,
        "base_px_y": base_px_y
    }
    
    with open("calibration.json", "w") as f:
        json.dump(cal_data, f, indent=4)
        
    print("\n=== Calibration Complete ===")
    print(f"Scale X: {scale_x:.4f} mm/px")
    print(f"Scale Y: {scale_y:.4f} mm/px")
    print(f"Calculated Robot Base Pixel Coordinate: ({base_px_x:.1f}, {base_px_y:.1f})")
    print("Saved spatial mappings to calibration.json! main.py will now use this data automatically.")

if __name__ == "__main__":
    main()
