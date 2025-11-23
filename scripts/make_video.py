import cv2
import numpy as np
import os

def create_scrolling_video(image_path, output_path, duration_sec=10, fps=30):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    h, w, c = img.shape
    
    # We simulate a camera window of 1024x1024 (or 16:9 ratio)
    window_w = int(h * 1.77) # 16:9 aspect ratio based on height
    if window_w > w: window_w = w
    
    # Create Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (window_w, h))
    
    total_frames = duration_sec * fps
    pixels_per_frame = w / total_frames # Speed of scroll
    
    print(f"Rendering video to {output_path}...")
    
    # Create a double-width image for easy slicing (seamless loop logic)
    # [Image][Image]
    double_img = np.concatenate([img, img], axis=1)
    
    for i in range(total_frames):
        start_x = int(i * pixels_per_frame)
        # Crop the window
        frame = double_img[:, start_x : start_x + window_w, :]
        out.write(frame)
        
    out.release()
    print("Video saved!")

if __name__ == "__main__":
    create_scrolling_video("../output/final_panorama.png", "../output/panorama_scroll.mp4")