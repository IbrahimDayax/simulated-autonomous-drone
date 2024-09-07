import airsim
import numpy as np

class DroneEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.action_space = 6  # Six possible actions: forward, backward, left, right, up, down
        self.state_space = 100  # Define an appropriate state space size
        self.target_position = np.array([302.93, 8.84])  # Target coordinates
        self.reset()

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        state = self._get_state()
        return self._discretize_state(state)

    def step(self, action):
        # Define the drone's movement based on the action
        z = -20  # Target altitude

        if action == 0:
            self.client.moveByVelocityZAsync(1, 0, z, 1).join()
        elif action == 1:
            self.client.moveByVelocityZAsync(-1, 0, z, 1).join()
        elif action == 2:
            self.client.moveByVelocityZAsync(0, 1, z, 1).join()
        elif action == 3:
            self.client.moveByVelocityZAsync(0, -1, z, 1).join()
        elif action == 4:
            self.client.moveByVelocityZAsync(0, 0, z - 1, 1).join()
        elif action == 5:
            self.client.moveByVelocityZAsync(0, 0, z + 1, 1).join()

        next_state = self._get_state()
        reward, done = self._compute_reward()
        return self._discretize_state(next_state), reward, done

    def _get_state(self):
        # Process images to define the state
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
        ])
        
        # Process the image data to create the state representation
        state = np.concatenate([
            np.array(response.image_data_float).flatten() if response.pixels_as_float else np.frombuffer(response.image_data_uint8, dtype=np.uint8).flatten()
            for response in responses
        ])
        return state

    def _discretize_state(self, state):
        state_bins = np.linspace(0, 1, self.state_space)  # Define the state bins
        discretized_state = np.digitize(state, state_bins) - 1  # Discretize the state values
        return discretized_state

    def _compute_reward(self):
        # Compute reward based on the drone's state
        collision_info = self.client.simGetCollisionInfo()
        position = self.client.getMultirotorState().kinematics_estimated.position
        current_position = np.array([position.x_val, position.y_val])

        if collision_info.has_collided:
            return -100, True

        # Check if the drone is near the target position
        distance_to_target = np.linalg.norm(current_position - self.target_position)
        if distance_to_target < 10:
            self.client.landAsync().join()  # Land the drone when it reaches the target
            return 100, True

        return -1, False  # Negative reward for each step taken

    def close(self):
        self.client.armDisarm(False)
        self.client.reset()
        self.client.enableApiControl(False)
