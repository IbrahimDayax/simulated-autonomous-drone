import airsim  # Import the AirSim client library for drone simulation
import numpy as np  # Import the NumPy library for numerical operations
import cv2  # Import the OpenCV library for image processing
import math  # Import the math library for mathematical operations
import time  # Import the time library for sleep functionality

# Function to set segmentation colors for specific objects in the simulation
def set_segmentation_colors():
    # Set segmentation ID for all objects matching the regex pattern [\w]* to 0 (default)
    client.simSetSegmentationObjectID("[\w]*", 0, True)
    # Set segmentation ID for the object named 'Landscape_1' to 0
    client.simSetSegmentationObjectID('Landscape_1', 0)
    # Set segmentation ID for the object named 'InstancedFoliageActor_0' to 1
    client.simSetSegmentationObjectID('InstancedFoliageActor_0', 1)

# Function to calculate the depth planner image
def calc_depth_planner_image():
    # Request the depth planner image from the simulation
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])
    response = responses[0]
    # Convert the image data to a 2D float array
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    # Flip the image vertically
    img2d = np.flipud(img2d)
    return img2d

# Function to calculate the depth visualization image
def calc_depth_vis_image():
    # Request the depth visualization image from the simulation
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])
    response = responses[0]
    # Convert the image data to a 2D float array
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    # Flip the image vertically
    img2d = np.flipud(img2d)
    return img2d

# Function to calculate the segmentation image
def calc_segmentation_image():
    # Request the segmentation image from the simulation
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
    response = responses[0]
    # Convert the image data to a 1D uint8 array
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    # Reshape the 1D array to a 3D RGB image
    img_rgb = img1d.reshape((response.height, response.width, 3))
    return img_rgb

# Main image processing algorithm
def image_proc_algorithm():
    # Calculate the depth planner image
    depth_planner_image = calc_depth_planner_image()
    # Calculate the depth visualization image
    depth_vis_image = calc_depth_vis_image()
    # Flip the depth visualization image vertically
    depth_vis_image = np.flipud(depth_vis_image)
    # Calculate the segmentation image
    segmentation_image = calc_segmentation_image()

    # Initialize an output image with the same shape as the depth visualization image
    out_ = np.ones(depth_vis_image.shape, dtype=depth_vis_image.dtype)
    
    # Iterate over each pixel in the depth visualization image
    for i in range(depth_vis_image.shape[0]):
        for j in range(depth_vis_image.shape[1]):
            # If the depth value is less than 0.2
            if depth_vis_image[i][j] < 0.2:
                # If the segmentation color matches the specified color
                if np.array_equal(segmentation_image[i][j], np.array([42, 174, 203])):
                    # Set the output pixel to 0 (obstacle)
                    out_[i][j] = 0.0
                else:
                    # Set the output pixel to 1 (free space)
                    out_[i][j] = 1.0
            else:
                # Set the output pixel to 1 (free space)
                out_[i][j] = 1.0

    # Flip the output image vertically
    out_ = np.flipud(out_)

    # Create a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    # Apply erosion to the output image
    erode_image = cv2.erode(out_, kernel)

    # Add a border to the eroded image
    erode_image = cv2.copyMakeBorder(erode_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    # Flip the eroded image vertically
    erode_image = np.flipud(erode_image)

    # Convert the eroded image to an 8-bit unsigned integer type
    seg = np.array(erode_image).astype(dtype=np.uint8)
    # Normalize the image to the range 0-255
    seg = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX)
    # Make a copy of the segmented image
    main = np.copy(seg)
    # Convert the grayscale image to a BGR image
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)

    # Apply the Canny edge detection algorithm to the segmented image
    seg = cv2.Canny(seg, 0, 255)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add a border to the output and depth planner images
    out_ = cv2.copyMakeBorder(out_, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    depth_planner_image = cv2.copyMakeBorder(depth_planner_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

    # Initialize variables for tracking the closest obstacle
    depth = float('inf')
    
    # Iterate over each contour
    for i, c in enumerate(contours):
        # Get the minimum area rectangle enclosing the contour
        rect = cv2.minAreaRect(c)
        # Get the corner points of the rectangle
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Use np.intp instead of np.int0

        # Calculate the area of the rectangle
        pos1, pos2, pos3, pos4 = box
        area = abs((pos1[0] * pos2[1] - pos1[1] * pos2[0]) + (pos2[0] * pos3[1] - pos2[1] * pos3[0]) +
                   (pos3[0] * pos4[1] - pos3[1] * pos4[0]) + (pos4[0] * pos1[1] - pos4[1] * pos1[0])) / 2
        
        # If the area is greater than 1000 pixels
        if area > 1000:
            # Get the minimum and maximum x and y coordinates of the rectangle
            min_x, max_x = (min([pos1[0], pos2[0], pos3[0], pos4[0]]), max([pos1[0], pos2[0], pos3[0], pos4[0]]))
            min_y, max_y = (min([pos1[1], pos2[1], pos3[1], pos4[1]]), max([pos1[1], pos2[1], pos3[1], pos4[1]]))
            
            # Adjust the coordinates to fit within the image dimensions
            if max_x >= np.shape(out_)[1]:
                max_x = np.shape(out_)[1]
            if max_y >= np.shape(out_)[0]:
                max_y = np.shape(out_)[0]
            
            depth = float('inf')
            # Iterate over the pixels within the bounding box of the rectangle
            for k in range(min_y, max_y):
                for j in range(min_x, max_x):
                    # If the pixel is an obstacle
                    if out_[k][j] == 0.0:
                        # Update the obstacle depth to the minimum depth value
                        if depth_planner_image[k, j] < depth:
                            depth = depth_planner_image[k, j]
                        break
            
            # Draw the bounding box of the obstacle on the main image
            cv2.drawContours(main, [box], 0, (0, 0, 255), 2)

    # Create a more comprehensive state representation
    depth_values = depth_planner_image.flatten()
    obstacle_values = out_.flatten()
    state = np.concatenate((depth_values, obstacle_values))

    return state  # Return the state representation

# Load the trained Q-table
q_table = np.load("../data/best_q_table.npy")

# Define the state size and action size
state_size = q_table.shape[0]
action_size = q_table.shape[1]

# Main algorithm for escaping and navigating through obstacles
def auto_nav_algorithm():
    try:
        print("Starting escape algorithm")
        client.confirmConnection()  # Confirm connection to the AirSim simulator
        client.enableApiControl(True)  # Enable API control
        client.armDisarm(True)  # Arm the drone
        set_segmentation_colors()  # Set segmentation colors for objects
        client.takeoffAsync().join()  # Take off the drone asynchronously

        z = -20  # Set the initial altitude
        duration = 1  # Set the duration for movement
        speed = 1  # Set the speed for movement

        target_x, target_y = 302.93, 8.84  # Define the target coordinates

        while True:
            state = image_proc_algorithm()  # Get the state representation from the image processing algorithm
            state_discretized = np.digitize(state, np.linspace(0, 1, state_size)) - 1  # Discretize state values

            action = np.argmax(q_table[state_discretized])  # Select the action with the highest Q-value

            if action == 0:
                client.moveByVelocityZAsync(speed, 0, z, duration).join()  # Move forward
            elif action == 1:
                client.moveByVelocityZAsync(-speed, 0, z, duration).join()  # Move backward
            elif action == 2:
                client.moveByVelocityZAsync(0, speed, z, duration).join()  # Move left
            elif action == 3:
                client.moveByVelocityZAsync(0, -speed, z, duration).join()  # Move right
            elif action == 4:
                z -= speed  # Move down
                client.moveByVelocityZAsync(0, 0, z, duration).join()
            elif action == 5:
                z += speed  # Move up
                client.moveByVelocityZAsync(0, 0, z, duration).join()

            x = client.getMultirotorState().kinematics_estimated.position.x_val  # Get the current x position of the drone
            y = client.getMultirotorState().kinematics_estimated.position.y_val  # Get the current y position of the drone
            print(x, y)  # Print the current position

            if abs(x - target_x) < 10 and abs(y - target_y) < 10:  # Check if the drone is within 10 units of the target
                client.landAsync().join()  # Land the drone asynchronously
                client.armDisarm(False)  # Disarm the drone
                client.reset()  # Reset the simulator
                client.enableApiControl(False)  # Disable API control
                print("Mission complete.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        client.armDisarm(False)  # Disarm the drone
        client.reset()  # Reset the simulator
        client.enableApiControl(False)  # Disable API control
    except Exception as e:
        print(f"An error occurred: {e}")
        client.armDisarm(False)  # Disarm the drone
        client.reset()  # Reset the simulator
        client.enableApiControl(False)  # Disable API control

# Main entry point
if __name__ == "__main__":
    client = airsim.MultirotorClient()  # Initialize the AirSim client
    auto_nav_algorithm()  # Start the autonomous navigation algorithm
