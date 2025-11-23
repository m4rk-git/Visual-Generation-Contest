import cv2
import numpy as np
import os

def get_perspective_map(w_out, h_out, w_in, h_in, fov_deg, theta_deg, phi_deg=0):
    """
    Generates the X and Y mapping for cv2.remap to convert an Equirectangular (Panorama)
    image into a Perspective (Camera) view.
    """
    # 1. Camera Intrinsics
    f = 0.5 * w_out / np.tan(0.5 * np.deg2rad(fov_deg))
    cx, cy = w_out / 2, h_out / 2

    # 2. Create Grid of Output Pixels
    x_rng = np.arange(0, w_out, dtype=np.float32) - cx
    y_rng = np.arange(0, h_out, dtype=np.float32) - cy
    x_grid, y_grid = np.meshgrid(x_rng, y_rng)

    # 3. Unproject to 3D Rays (P_camera)
    # Coordinate system: Z is forward, X is right, Y is down
    z_grid = np.full_like(x_grid, f)
    xyz = np.stack([x_grid, y_grid, z_grid], axis=-1) # Shape: [H, W, 3]
    
    # Normalize rays
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz = xyz / norm

    # 4. Rotate Rays (Camera Rotation)
    # Rotation around Y-axis (Horizontal Pan) - Theta
    theta = np.deg2rad(theta_deg)
    # Rotation around X-axis (Vertical Tilt) - Phi
    phi = np.deg2rad(phi_deg)

    # Rotation Matrices
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    
    # Apply Rotation: R = Ry @ Rx
    R = Ry @ Rx
    
    # Rotate the rays: ray_world = ray_cam @ R.T
    # We use tensordot for efficient multiplication over the grid
    xyz_rotated = np.tensordot(xyz, R.T, axes=1) 

    # 5. Convert 3D Rays to Spherical Coordinates (Longitude/Latitude)
    x_r, y_r, z_r = xyz_rotated[..., 0], xyz_rotated[..., 1], xyz_rotated[..., 2]
    
    # Longitude (lambda) = atan2(x, z)
    lon = np.arctan2(x_r, z_r)
    # Latitude (phi) = asin(y / 1.0) -> since we normalized, vector length is 1
    # Note: In panoramas, Y is usually down. 
    lat = np.arcsin(y_r)

    # 6. Map Spherical to UV (Image Coordinates)
    # Longitude: [-pi, pi] -> [0, W_in]
    # We need to handle the wrapping (modulo) for seamlessness
    u = (lon / (2 * np.pi) + 0.5) * w_in
    
    # Latitude: [-pi/2, pi/2] -> [0, H_in]
    v = (lat / np.pi + 0.5) * h_in

    # 7. Wraparound Logic (Seamless)
    # If u goes slightly negative or beyond W, wrap it
    u = np.mod(u, w_in)
    
    return u.astype(np.float32), v.astype(np.float32)

def create_immersive_video(image_path, output_path, duration_sec=12, fps=30, fov=90):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return
    
    h_in, w_in, c = img.shape
    
    # Output Resolution (Standard HD)
    w_out, h_out = 1920, 1080
    
    print(f"Processing Immersive Video ({w_out}x{h_out})...")
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))
    
    total_frames = duration_sec * fps
    
    # Pre-calculate the base grid to speed up the loop? 
    # No, fast enough to do per frame for 10 seconds.
    
    for i in range(total_frames):
        # Calculate angle (0 to 360 degrees)
        theta = (i / total_frames) * 360.0
        
        # Generate the mapping for this angle
        # We define -theta so it rotates the correct way (pan right)
        map_x, map_y = get_perspective_map(w_out, h_out, w_in, h_in, fov, -theta)
        
        # Remap (Warps the image)
        # INTER_LINEAR is fast and looks good. INTER_CUBIC is sharper but slower.
        frame = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        out.write(frame)
        
        if i % 30 == 0:
            print(f"Rendered frame {i}/{total_frames}")
            
    out.release()
    print(f"Immersive video saved to {output_path}")

if __name__ == "__main__":
    # Input should be your large generated panorama
    input_img = "../output/final_panorama.png"
    output_vid = "../output/final_immersive_view.mp4"
    create_immersive_video(input_img, output_vid)