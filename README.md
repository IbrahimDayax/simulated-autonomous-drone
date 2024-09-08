# Simulated Autonomous Drone Navigation Using Q-Learning and Computer Vision

This project successfully developed and evaluated autonomous navigation systems for
drones using three different approaches: computer vision, reinforcement learning (Q-Learning),
and a hybrid method combining both. The computer vision system, though capable of
detecting and navigating around obstacles, exhibited longer flight times and higher
collision rates. The reinforcement learning model demonstrated significant
improvement in efficiency and collision avoidance, albeit with a more complex training
process. The hybrid approach effectively leveraged the strengths of both methods,
achieving the shortest flight times and minimal collisions, thus demonstrating the
potential of integrating multiple techniques for enhanced autonomous navigation. The
training process involved rigorous simulation and evaluation, leading to a robust final
model.

## Requirements

The project was simulated using Unreal Engine 4.27 and AirSim. We also used various Python libraries for implementing machine learning algorithms and performing computer vision tasks. These packages can be installed using Conda or pip. I used the AirSim Forest environment which can be found and downloaded from this [link](https://github.com/microsoft/AirSim/releases/tag/v.1.2.2). I chose to use the forest environment for both the variety of obstacles in forest environments and the practical use cases of having an autonomous drone capable of navigating a forest while avoiding obstacles for search and resecue missions as well as exploration and fire watching.

### Environment Setup

An environment.yml file is included to install all necessary dependencies.

### Dependencies

- Python 3.8.19
- Numpy 1.24.4 - Array handling and numerical operations
- OpenCV 4.10.0.82 - Computer vision library for obstacle detection
- Airsim 1.2.2 - Unreal Engine plugin for drone simulation
- Msgpack - Message serialization for efficient data exchange
- Tornado - Web framework for real-time communication with the drone

## Installation

1. Clone the repository:
   - ```bash
     git clone https://github.com/IbrahimDayax/simulated-autonomous-drone.git
2. Navigate to the project directory:
   - ```bash
     cd simulated-autonomous-drone
3. Create a Conda environment from the provided environment.yml file:
   - ```bash
     conda env create -f environment.yml
4. Activate the environment:
   - ```bash
     conda activate autonomous_drone_navigation

## Simulation Setup

In this project, the AirSim plugin, built on Unreal Engine 4, was used to simulate autonomous UAV navigation and obstacle detection in a high-fidelity test environment. The AirSim forest environment, rich in obstacles, was chosen for its realism, making it ideal for testing drone navigation in forest-like scenarios. The UAV was equipped with three primary cameras: a Depth Camera for generating depth maps for obstacle avoidance, a Segmentation Camera for distinguishing objects from the background, and an FPV Camera to simulate the pilot's view. The system provided comprehensive visual feedback, including a bird's-eye view for tracking the UAV's location during flight.
![screen01](https://github.com/user-attachments/assets/f69973ff-92f0-4c1a-94a4-8dd04df77ea1)


## System Architecture

### Q-Learning

Q-learning is used to enable the drone to learn an optimal policy for reaching its destination by trial and error. The algorithm updates its Q-values based on rewards received from the environment for every action taken. Over time, the agent learns the best set of actions to take to maximize cumulative rewards, avoiding obstacles and minimizing flight time.

- State Representation: The drone's position, velocity, and proximity to obstacles.
- Action Space: Possible movements in 3D space (up, down, left, right, forward, backward).
- Reward System: Positive rewards for moving towards the goal and negative rewards for collisions with obstacles.

