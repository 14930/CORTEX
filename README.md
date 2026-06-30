# 🤖 CORTEX: KUKA Robot Cable Tracking & Reinforcement Learning Environment

> **A PyBullet-based simulation environment for robotic cable manipulation, integrating Computer Vision (Open3D) and Reinforcement Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#-technology-stack)
[![PyBullet](https://img.shields.io/badge/Physics-PyBullet-orange)](#-technology-stack)
[![OpenCV & Open3D](https://img.shields.io/badge/Vision-OpenCV%20%26%20Open3D-green)](#-technology-stack)

CORTEX is a dynamic robotic simulation workspace tailored for complex cable manipulation tasks using a KUKA LBR iiwa robotic arm. It provides a robust physics environment with full synthetic camera streams (RGB, Depth, Segmentation), bridging the gap between Computer Vision (Member 4) and Reinforcement Learning (Member 3).

---

## 🎯 The Goal

Manipulating deformable objects like cables is an open challenge in robotics. CORTEX provides an end-to-end framework where:
1. **Vision Engine:** Synthetic depth cameras segment the cable, extract its 3D Point Cloud, and track the highly dynamic free-floating tip.
2. **RL Agent:** Learns to actuate the 7-DOF KUKA arm to navigate the cable tip precisely to a designated target position in 3D space.

## 🚀 Key Features

### 📡 Real-Time Cable Vision & Tracking
Using PyBullet's synthetic camera matrices, CORTEX extracts real-time depth and segmentation data. The vision module (`detectigCable.py`) reconstructs this into a 3D Point Cloud using **Open3D** and tracks the exact 3D coordinates of the cable tip.

<p align="center">
  <img src="kuka_tracking_demo.png" alt="KUKA PyBullet Simulation and Computer Vision Tracking" width="850">
</p>

### 🦾 PyBullet Physics & Deformables
The cable is modeled using interconnected dynamic rigid bodies with tuned stiffness constraints and multi-body dynamics, simulating real-world sag and momentum.

### 🧠 Reinforcement Learning Ready (Gym API)
The `KukaCableGymEnv` class exposes a standard step-reward-reset API. It calculates state observations (joint angles + cable topology) and shapes rewards based on the Euclidean distance between the dynamic cable tip and the target.

---

## 🛠 Technology Stack

| Component | Technologies |
|---|---|
| **Physics Simulator** | PyBullet |
| **Robotic Arm** | KUKA LBR iiwa (7 DOF) |
| **Computer Vision** | OpenCV, NumPy, Matplotlib |
| **3D Point Clouds** | Open3D |
| **Environment Interface** | Custom Gym-like API (`kuka_env.py`) |

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Setup the Environment
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-username/CORTEX.git
cd CORTEX
pip install -r requirements.txt
```

*(Optional but recommended: Use a virtual environment)*
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Vision Tracking Demo
Launch the interactive tracking simulation which streams live point cloud tracking to CSV:
```bash
python detectigCable.py
```
This will open the simulation, display the masked vision feed, and save the trajectory to `cable_tracking_data.csv`.

---

## 🚀 Roadmap (Next Phases)
- **Phase 1 (Complete):** Physics Environment, Cable Physics, and Vision Point Cloud Extraction.
- **Phase 2 (Upcoming):** **Reinforcement Learning (RL) Pipeline**. Training an agent (e.g., PPO/SAC) to dynamically swing the cable tip to the target by learning the underlying non-linear cable dynamics. 

## 📜 License
This project is licensed under the MIT License.
