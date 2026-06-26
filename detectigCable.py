from kuka_env import KukaCableGymEnv
import numpy as np
import cv2
import open3d as o3d

# ── Helper Functions ──────────────────────────────────────────
def depth_to_3d(depth_array, seg_array, cable_ids, fov=65, width=640, height=480):
    fx = width / (2 * np.tan(np.radians(fov / 2)))
    fy = fx

    cable_mask = np.isin(seg_array, cable_ids)
    ys, xs = np.where(cable_mask)

    positions = []
    for x, y in zip(xs, ys):
        z = depth_array[y, x]
        x3d = (x - width / 2) * z / fx
        y3d = (y - height / 2) * z / fy
        positions.append([x3d, y3d, z])

    return np.array(positions)


def create_point_cloud(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.paint_uniform_color([0, 1, 0])
    return pcd


# ── Main Loop ─────────────────────────────────────────────────
env = KukaCableGymEnv(use_gui=True)
obs = env.reset()

for step in range(100):
    rgb, depth, seg = env.get_full_camera_data()

    # Isolate cable visually
    cable_mask = np.isin(seg, env.cable_segments)
    cable_visual = rgb.copy()
    cable_visual[~cable_mask] = 0
    cv2.imshow("Cable Tracking", cable_visual)
    cv2.waitKey(1)

    # Get 3D positions
    cable_points_3d = depth_to_3d(depth, seg, env.cable_segments)

    if len(cable_points_3d) > 0:
        pcd = create_point_cloud(cable_points_3d)
        print(f"Step {step} | Cable points detected: {len(cable_points_3d)}")
    else:
        print(f"Step {step} | No cable detected in frame")

    # Random action (Member 3 will replace with RL agent later)
    action = np.zeros(7)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()

env.close()
cv2.destroyAllWindows()


# ── What you send to Member 3 ─────────────────────────────────
if len(cable_points_3d) > 0:
    cable_tip = cable_points_3d[-1]          # [x, y, z] of cable end
    enhanced_obs = np.append(obs, cable_tip) # adds your 3 values to their obs
else:
    cable_tip = np.zeros(3)
    enhanced_obs = np.append(obs, cable_tip)

print(f"Step {step} | Cable Tip: {cable_tip} | Obs size: {len(enhanced_obs)}")


