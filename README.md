# dora-sim-genesis-franka-grasp-cube

This Dora node launches a simulated environment featuring a Franka robot and a graspable cube. It publishes real-time data streams—including the robot’s joint states and end-effector pose—and accepts incoming control commands to drive the robot’s motion.

## Installation

You can install the package in development mode using either **uv** or **conda**:

```bash
pip install -e .
```

To enable visualization support (e.g., with Rerun), install the optional `view` extras:

```bash
pip install -e ".[view]"
```

## Usage

After installation, the node can be launched as part of a Dora dataflow. Visualization of the simulation scene is available when the `view` extra is installed.
