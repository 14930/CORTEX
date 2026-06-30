from kuka_env import KukaCableGymEnv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    print("Initializing PyBullet environment for screenshots...")
    # use_gui=False because we don't need the pybullet GUI window, we'll just save the camera buffer
    env = KukaCableGymEnv(use_gui=False)
    env.reset()
    
    # Let it settle or move a bit
    for _ in range(10):
        env.step(np.zeros(7))
        
    rgb, depth, seg = env.get_full_camera_data()
    
    # Save raw RGB from simulation
    # PyBullet RGB comes as RGBA, converting to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("kuka_sim_rgb.png", bgr)
    
    # Create the cable visual mask
    cable_mask = np.isin(seg, env.cable_segments)
    cable_visual = rgb.copy()
    cable_visual[~cable_mask] = 0
    bgr_mask = cv2.cvtColor(cable_visual, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("kuka_sim_mask.png", bgr_mask)
    
    # Create a nice composite visualization (Simulation vs Computer Vision Tracking)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("PyBullet Simulation (RGB)", fontsize=14)
    axes[0].axis("off")
    
    axes[1].imshow(cable_visual)
    axes[1].set_title("Computer Vision Cable Tracking", fontsize=14)
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("kuka_tracking_demo.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved images successfully.")
    env.close()

if __name__ == "__main__":
    main()
