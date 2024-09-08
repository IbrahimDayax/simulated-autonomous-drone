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

1. 
    ```bash
