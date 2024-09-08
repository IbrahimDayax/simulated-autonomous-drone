# Simulated Autonomous Drone Navigation Using Reinforcement Learning (Q-Learning) and Computer Vision

This project simulates autonomous drone navigation using a combination of Q-Learning and computer vision techniques within the AirSim environment based on Unreal Engine 4. The drone, equipped with Depth, Segmentation, and FPV cameras, navigates a forest-like environment while avoiding obstacles to reach and land on given target coordinates. The system integrates computer vision for real-time obstacle detection and Q-Learning for decision-making, resulting in a hybrid approach that significantly improves navigation efficiency and collision avoidance compared to using either method alone.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Simulation Setup](#simulation-setup)
- [Project Files](#project-files)
- [Detailed Overview of the Different Methods Used](#Detailed-Overview-of-the-Different-Methods-Used)
- [Results](#results)
- [Q-Learning Model Performance](#Q-Learning-Model-Performance)

## Requirements

1. Conda
2. Unreal Engine 4.27 & AirSim
3. Laptop with good cpu and ram (minimum core i5 with 16gb ram)

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
3. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)  
4. Create a Conda environment from the provided environment.yml file:
   - ```bash
     conda env create -f environment.yml
5. Activate the environment:
   - ```bash
     conda activate autonomous_drone_navigation
6. Install [Unreal Engine 4.27 and AirSim](https://microsoft.github.io/AirSim/build_windows/)
7. Download the AirSim [forest](https://github.com/microsoft/AirSim/releases/tag/v.1.2.2) environment then extract and then run it's exe file
8. To test the Q-Learning model run
   - ```bash
     python test.py
   and to train it further run
   - ```bash
     python train.py
9. To simulate the Hybrid combined CV-RL method run
   - ```bash
     python hybrid_auto_drone.py
10. To simulate the Computer Vision without RL method run
    - ```bash
      python cv_auto_drone.py
      
## Simulation Setup

In this project, the AirSim plugin, built on Unreal Engine 4, was used to simulate autonomous UAV navigation and obstacle detection in a high-fidelity test environment. The AirSim forest environment, rich in obstacles, was chosen for its realism, making it ideal for testing drone navigation in forest-like scenarios. The UAV was equipped with three primary cameras: a Depth Camera for generating depth maps for obstacle avoidance, a Segmentation Camera for distinguishing objects from the background, and an FPV Camera to simulate the pilot's view. The system provided comprehensive visual feedback, including a bird's-eye view for tracking the UAV's location during flight.

![screen05](https://github.com/user-attachments/assets/b55c3fac-ab15-4eb6-b405-2809b7da1558)

## Project Files

Here is an overview of the files in this project and their purposes:

### Configuration Files

- **settings.json:** This file configures the AirSim simulation environment. It specifies the simulation mode as "Multirotor" and sets the view mode to "FlyWithMe". The file configures a drone with three cameras—DepthCamera, SegmentationCamera, and FPVCamera—each with specified capture settings. It also defines sub-windows for displaying images from these cameras during the simulation. The logging settings are set to verbose, allowing detailed message logs.

### Code Files

1. **cv_auto_drone.py:** This script processes images from the drone's cameras to detect and avoid obstacles. It uses segmentation and depth images to identify and measure obstacles, applying OpenCV techniques to find and track these obstacles. This information helps the drone navigate safely around obstacles while moving towards its goal.
2. **environment.py:** This file is the environment setup script. It defines a DroneEnv class for controlling a drone in AirSim. It handles drone initialization, movement based on actions, state updates using image data, and reward calculation for navigation and collision avoidance. The step method moves the drone and computes rewards, while the close method stops and resets the drone.
3. **hybrid_auto_drone.py:** This script processes depth and segmentation images to detect and avoid obstacles. It uses OpenCV to identify obstacles, update the drone's state, and guide its navigation. A Q-learning model then determines the drone's actions to navigate safely towards its target.
4. **q_learning_agent.py:** This file defines a QLearningAgent class for implementing a Q-learning algorithm. It initializes a Q-table, manages exploration-exploitation trade-offs with an epsilon-greedy strategy, and updates the Q-table based on learning from rewards and next states. The drone agent can choose actions, learn from experiences, and save or load the Q-table to/from a file.
5. **train.py:** This script handles the training process of the Q-learning agent. It runs the simulation for several episodes, logs performance metrics, saves the trained Q-table or rewards and reports the best-performing episode.
6. **test.py:** This script tests the Q-learning agent's performance by loading the best Q-table and evaluating the drone agent in the DroneEnv environment over a specified number of episodes. It prints the total reward for each test episode and then closes the environment.

### Data Directory Files

1. **best_q_table.npy:** This file stores the best Q-table found during training. The Q-table contains learned values for each state-action pair and represents the agent’s knowledge of optimal actions based on prior experience.
2. **best_reward.npy:** This file saves the best reward achieved during training, which represents the highest cumulative reward the agent obtained in an episode. It’s useful for monitoring progress and evaluating the efficiency of the learned policy.
3. **last_episode.npy:** This file stores the state of the Q-table, rewards, or other metrics from the last training episode. It allows you to resume training from where you left off or analyze the performance at the final stage of training.
4. **q_table.npy:** This file contains the current Q-table, which is used by the agent to determine the best actions during training or testing. It stores the values for all state-action pairs and is updated as the agent interacts with the environment.
5. **total_rewards.npy:** This file logs the total rewards obtained across all episodes. It helps track the performance of the agent over time, showing whether the learning algorithm is improving and how quickly it converges to an optimal solution.

All these files work together to define the environment, train the Q-learning agent, and store performance data that can be used to improve or test the system.

## Detailed Overview of the Different Methods Used

### Using Only Computer Vision & Image Processing

The computer vision-based algorithm enables autonomous drone navigation and real-time obstacle avoidance in complex environments. The drone, simulated in AirSim, utilizes three primary cameras: Depth, Segmentation, and FPV. The Depth Camera generates depth maps to gauge object distances, while the Segmentation Camera distinguishes between objects using color-coded segmentation. FPV provides a standard visual feed. The algorithm processes depth and segmentation data to detect obstacles, applying techniques like Canny edge detection to refine object boundaries. The drone continuously estimates distances to obstacles, adjusting its path when objects are detected within a 5-meter range, ensuring collision-free navigation toward the target.

This approach was implemented in the **cv_auto_drone.py** script, where image processing and computer vision were used to enable the drone to detect and avoid obstacles in its environment. The drone captures segmentation and depth images using the AirSim environment. Segmentation images help classify objects by assigning different colors, while depth images provide distance information from the drone to objects. OpenCV is then used for processing these images by detecting contours, applying thresholding, and calculating bounding boxes around obstacles. These bounding boxes help determine the location of the closest obstacle relative to the drone, allowing it to choose safe escape maneuvers and avoid collisions. The processed images guide the drone’s navigation logic to ensure it moves toward its goal while avoiding obstacles.

### Q-Learning

Q-learning is used to enable the drone to learn an optimal policy for reaching its destination by trial and error. The algorithm updates its Q-values based on rewards received from the environment for every action taken. Over time, the agent learns the best set of actions to take to maximize cumulative rewards, avoiding obstacles and minimizing flight time.

- **State Representation:** The drone's position, velocity, and proximity to obstacles.
- **Action Space:** Possible movements in 3D space (up, down, left, right, forward, backward).
- **Reward System:** Positive rewards for moving towards the goal and negative rewards for collisions with obstacles.

![screen05](https://github.com/user-attachments/assets/1324f10b-d5fc-4bfe-acff-99e3987c3745)

### Hybrid Computer Vision & Q-Learning

The hybrid approach combines the strengths of Q-learning's decision-making capabilities with computer vision's obstacle detection. The system uses the camera feed to detect obstacles and then uses Q-learning to plan the optimal route to avoid collisions while heading toward the goal. This hybrid method is implemented in the **hybrid_auto_drone.py** script where image processing and computer vision play a crucial role in the drone's autonomous navigation and obstacle avoidance. The script captures depth and segmentation images from the AirSim environment. Depth images provide distance information, and segmentation images differentiate objects based on their assigned colors. OpenCV is used to process these images, detecting obstacles by analyzing pixel depth and applying morphological operations like erosion. The detected obstacles are enclosed within bounding boxes, and the nearest one is identified based on its depth. These obstacles are used to update the drone's state, guiding it in choosing escape maneuvers to avoid collisions and reach its target. The script then employs a trained Q-learning model to select the appropriate action based on the processed state data, enabling the drone to navigate toward its target coordinates autonomously.


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


