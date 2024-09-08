# Simulated Autonomous Drone Navigation Using Q-Learning and Computer Vision

This project successfully developed and evaluated autonomous navigation systems for drones using three different approaches: computer vision, reinforcement learning (Q-Learning), and a hybrid method combining both. The computer vision system, though capable of detecting and navigating around obstacles, exhibited longer flight times and higher collision rates. The reinforcement learning model demonstrated significant improvement in efficiency and collision avoidance, albeit with a more complex training
process. The hybrid approach effectively leveraged the strengths of both methods, achieving the shortest flight times and minimal collisions, thus demonstrating the potential of integrating multiple techniques for enhanced autonomous navigation. The training process involved rigorous simulation and evaluation, leading to a robust final model.

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

![screen05](https://github.com/user-attachments/assets/b55c3fac-ab15-4eb6-b405-2809b7da1558)


## System Architecture

### Using Computer Vision alone

The computer vision-based algorithm enables autonomous drone navigation and real-time obstacle avoidance in complex environments. The drone, simulated in AirSim, utilizes three primary cameras: Depth, Segmentation, and FPV. The Depth Camera generates depth maps to gauge object distances, while the Segmentation Camera distinguishes between objects using color-coded segmentation. FPV provides a standard visual feed. The algorithm processes depth and segmentation data to detect obstacles, applying techniques like Canny edge detection to refine object boundaries. The drone continuously estimates distances to obstacles, adjusting its path when objects are detected within a 5-meter range, ensuring collision-free navigation toward the target.

### Q-Learning

Q-learning is used to enable the drone to learn an optimal policy for reaching its destination by trial and error. The algorithm updates its Q-values based on rewards received from the environment for every action taken. Over time, the agent learns the best set of actions to take to maximize cumulative rewards, avoiding obstacles and minimizing flight time.

- **State Representation:** The drone's position, velocity, and proximity to obstacles.
- **Action Space:** Possible movements in 3D space (up, down, left, right, forward, backward).
- **Reward System:** Positive rewards for moving towards the goal and negative rewards for collisions with obstacles.

![screen05](https://github.com/user-attachments/assets/1324f10b-d5fc-4bfe-acff-99e3987c3745)

### Hybrid Computer Vision & Q-Learning

The hybrid approach combines the strengths of Q-learning's decision-making capabilities with computer vision's obstacle detection. The system uses the camera feed to detect obstacles and then uses Q-learning to plan the optimal route to avoid collisions while heading toward the goal.

## Results

### Performance Comparison Between the 3 Methods

In our evaluation, we compared three different approaches: computer vision, Q-learning, and the hybrid method. Below is a table summarizing the key metrics:

![screen05](https://github.com/user-attachments/assets/d7ecd611-a40c-4f77-a96a-791b4891a178)

- **Q-Learning** alone performed well in terms of obstacle avoidance but was slower than the hybrid method.
- Using **Computer Vision without Q-Learning** provided better obstacle detection but was prone to collisions due to dynamic changes in the environment.
- The **Hybrid CV-RL** approach yielded the best performance, with fewer steps taken and no collisions, outperforming both standalone methods.

### Q-Learning Model Performance 

#### Graph 1: Total Steps Taken to Reach Target

![screen01](https://github.com/user-attachments/assets/e2fdd6e5-a11f-479f-bc5b-c80fb5d53325)

This graph illustrates the Q-learning model's improvement in navigation efficiency over 100 episodes. As the training progresses, the total number of steps required to reach the target decreases significantly from approximately 500 steps to below 200 by episode 100. This trend demonstrates the system's learning capability and its ability to optimize its path as it gains experience.

#### Graph 2: Total Reward Per Episode

![screen04](https://github.com/user-attachments/assets/f56b8c3c-3ba1-4436-b77c-e79d5ddd032a)

The above provided graph displays the Q-learning model's performance in terms of total reward per episode over 100 training episodes. Initially, the total reward starts at around -400, indicating poor performance. As training progresses, the total reward increases steadily, reaching approximately -100 by the 75th episode and stabilizing around this value through to the 100th episode. This upward trend signifies that the agent is learning and improving its policy, gradually reducing the penalties incurred during training. The negative total reward throughout the episodes can be attributed to the reward structure, where the drone receives a -1 penalty for each step taken towards the target to incentivize the drone to be more efficient and a -100 penalty for collisions, with a +100 reward for reaching the target. The persistent negative total reward reflects these penalties, indicating that while the agent is improving, it still incurs penalties due to the steps taken to reach the target and occasional collisions.


