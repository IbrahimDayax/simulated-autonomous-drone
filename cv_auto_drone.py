import airsim  # Import the AirSim client library for drone simulation
import numpy as np  # Import the NumPy library for numerical operations
import cv2  # Import the OpenCV library for image processing
import math  # Import the math library for mathematical operations
import time  # Import the time library for sleep functionality

# Function to set segmentation colors for specific objects in the simulation
def set_segmentation_colors():
    client.simSetSegmentationObjectID("[\w]*", 0, True)  # Set segmentation ID for all objects matching the regex pattern to 0 (default)
    client.simSetSegmentationObjectID('Landscape_1', 0)  # Set segmentation ID for 'Landscape_1' to 0
    client.simSetSegmentationObjectID('InstancedFoliageActor_0', 1)  # Set segmentation ID for 'InstancedFoliageActor_0' to 1

# Function to calculate the depth planner image
def calc_depth_planner_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])  # Request depth planner image
    response = responses[0]  # Get the first response
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)  # Convert image data to 2D float array
    img2d = np.flipud(img2d)  # Flip the image vertically
    return img2d  # Return the processed image

# Function to calculate the depth visualization image
def calc_depth_vis_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])  # Request depth visualization image
    response = responses[0]  # Get the first response
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)  # Convert image data to 2D float array
    img2d = np.flipud(img2d)  # Flip the image vertically
    return img2d  # Return the processed image

# Function to calculate the segmentation image
def calc_segmentation_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])  # Request segmentation image
    response = responses[0]  # Get the first response

    # Check if the image response is valid
    if response.width == 0 or response.height == 0:
        print("Received an invalid image from AirSim")  # Print error message if invalid image
        return None  # Return None to indicate failure

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # Convert image data to 1D uint8 array

    # Check if the image data size matches the expected size
    if img1d.size != response.width * response.height * 3:
        print(f"Image data size mismatch. Expected: {response.width * response.height * 3}, Got: {img1d.size}")  # Print error message if size mismatch
        return None  # Return None to indicate failure

    img_rgb = img1d.reshape((response.height, response.width, 3))  # Reshape 1D array to 3D RGB image
    return img_rgb  # Return the processed image

# Main image processing algorithm
def image_proc_algorithm():
    depth_planner_image = calc_depth_planner_image()  # Calculate depth planner image
    depth_vis_image = calc_depth_vis_image()  # Calculate depth visualization image
    depth_vis_image = np.flipud(depth_vis_image)  # Flip depth visualization image vertically
    segmentation_image = calc_segmentation_image()  # Calculate segmentation image

    if segmentation_image is None:
        return float('inf'), None, None  # Return if segmentation image is invalid

    out_ = np.ones(depth_vis_image.shape, dtype=depth_vis_image.dtype)  # Initialize output image with ones (free space)

    # Iterate over each pixel in the depth visualization image
    for i in range(depth_vis_image.shape[0]):
        for j in range(depth_vis_image.shape[1]):
            # Check if the depth value indicates proximity to an obstacle
            if depth_vis_image[i][j] < 0.2:
                # Check if the segmentation color matches the obstacle color
                if np.array_equal(segmentation_image[i][j], np.array([42, 174, 203])):
                    out_[i][j] = 0.0  # Mark as obstacle
                else:
                    out_[i][j] = 1.0  # Mark as free space
            else:
                out_[i][j] = 1.0  # Mark as free space

    out_ = np.flipud(out_)  # Flip the output image vertically
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))  # Create structuring element for morphological operations
    erode_image = cv2.erode(out_, kernel)  # Apply erosion to the output image

    erode_image = cv2.copyMakeBorder(erode_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)  # Add border to the eroded image
    erode_image = np.flipud(erode_image)  # Flip the eroded image vertically

    seg = np.array(erode_image).astype(dtype=np.uint8)  # Convert eroded image to 8-bit unsigned integer type
    seg = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the image to the range 0-255
    main = np.copy(seg)  # Make a copy of the segmented image
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)  # Convert the grayscale image to a BGR image

    seg = cv2.Canny(seg, 0, 255)  # Apply Canny edge detection algorithm to the segmented image

    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the edge-detected image

    out_ = cv2.copyMakeBorder(out_, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)  # Add border to the output image
    depth_planner_image = cv2.copyMakeBorder(depth_planner_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)  # Add border to the depth planner image

    depth = float('inf')  # Initialize variables for tracking the closest obstacle
    closest_obstacle = None  # Initialize variable for the closest obstacle

    # Iterate over each contour
    for i, c in enumerate(contours):
        rect = cv2.minAreaRect(c)  # Get the minimum area rectangle enclosing the contour
        box = cv2.boxPoints(rect)  # Get the corner points of the rectangle
        box = np.intp(box)  # Convert the points to integer type

        pos1, pos2, pos3, pos4 = box  # Get the corner points
        area = abs((pos1[0] * pos2[1] - pos1[1] * pos2[0]) + (pos2[0] * pos3[1] - pos2[1] * pos3[0]) +
                   (pos3[0] * pos4[1] - pos3[1] * pos4[0]) + (pos4[0] * pos1[1] - pos4[1] * pos1[0])) / 2  # Calculate the area of the rectangle

        # Check if the area is larger than a threshold
        if area > 1000:
            # Get the minimum and maximum x and y coordinates of the rectangle
            min_x, max_x = (min([pos1[0], pos2[0], pos3[0], pos4[0]]), max([pos1[0], pos2[0], pos3[0], pos4[0]]))
            min_y, max_y = (min([pos1[1], pos2[1], pos3[1], pos4[1]]), max([pos1[1], pos2[1], pos3[1], pos4[1]]))

            # Adjust the coordinates to fit within the image dimensions
            if max_x >= np.shape(out_)[1]:
                max_x = np.shape(out_)[1]
            if max_y >= np.shape(out_)[0]:
                max_y = np.shape(out_)[0]

            obstacle_depth = float('inf')  # Initialize the obstacle depth to infinity

            # Iterate over the pixels within the bounding box of the rectangle
            for k in range(min_y, max_y):
                for j in range(min_x, max_x):
                    # Check if the pixel is an obstacle
                    if out_[k][j] == 0.0:
                        # Update the obstacle depth to the minimum depth value
                        if depth_planner_image[k, j] < obstacle_depth:
                            obstacle_depth = depth_planner_image[k, j]
                        break

            # Check if the obstacle depth is less than the current closest depth
            if obstacle_depth < depth:
                depth = obstacle_depth  # Update the closest depth
                closest_obstacle = box  # Update the closest obstacle

            cv2.drawContours(main, [box], 0, (0, 0, 255), 2)  # Draw the bounding box of the obstacle on the main image

    return depth, closest_obstacle, main  # Return the depth to the closest obstacle, the closest obstacle, and the main image

# Main algorithm for escaping and navigating through obstacles
def auto_nav_algorithm():
    try:
        print("Starting autonomous navigation algorithm")  # Print start message
        client.confirmConnection()  # Confirm connection to the AirSim simulator
        client.enableApiControl(True)  # Enable API control
        client.armDisarm(True)  # Arm the drone
        set_segmentation_colors()  # Set segmentation colors for objects
        client.takeoffAsync().join()  # Take off the drone asynchronously and wait until it reaches the takeoff altitude

        z = -20  # Set the initial altitude
        duration = 5  # Set the duration for each movement command
        speed = 1  # Set the initial speed of the drone

        target_position = np.array([302.93, 8.84])  # Set the target position

        while True:
            depth, closest_obstacle, _ = image_proc_algorithm()  # Get the depth and closest obstacle using the image processing algorithm
            print(f"Depth to closest obstacle: {depth}")  # Print the depth to the closest obstacle

            # Check if the depth to the closest obstacle is less than 5 meters
            if depth < 5:
                # Check if there is a detected closest obstacle
                if closest_obstacle is not None:
                    vx, vy, vz, yaw = 0, 0, 0, 0  # Initialize movement velocities and yaw angle
                    center_x = (closest_obstacle[0][0] + closest_obstacle[2][0]) / 2  # Get the center x-coordinate of the closest obstacle
                    center_y = (closest_obstacle[0][1] + closest_obstacle[2][1]) / 2  # Get the center y-coordinate of the closest obstacle

                    # Determine movement direction based on obstacle position
                    if center_x < 0.33 * np.shape(closest_obstacle)[0]:  # Obstacle is on the left
                        vy = speed
                        yaw = 90
                    elif center_x > 0.66 * np.shape(closest_obstacle)[0]:  # Obstacle is on the right
                        vy = -speed
                        yaw = -90
                    else:  # Obstacle is in front
                        if center_y < 0.5 * np.shape(closest_obstacle)[1]:  # Obstacle is above
                            vz = -speed
                        else:  # Obstacle is below
                            vz = speed

                    print(f"Moving by velocity vx={vx}, vy={vy}, vz={vz}, yaw={yaw}")  # Print the movement command
                    client.moveByVelocityZAsync(vx, vy, z + vz, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, yaw))  # Move the drone based on calculated velocities and yaw
                    time.sleep(1)  # Wait for 1 second
            else:
                position = client.getMultirotorState().kinematics_estimated.position  # Get the current position of the drone
                current_position = np.array([position.x_val, position.y_val])  # Extract the x and y coordinates
                print(current_position[0], current_position[1], position.z_val)  # Print the current position

                # Check if the drone has reached the target position
                if np.linalg.norm(current_position - target_position) < 5:
                    client.landAsync().join()  # Land the drone asynchronously
                    client.armDisarm(False)  # Disarm the drone
                    client.reset()  # Reset the simulator
                    client.enableApiControl(False)  # Disable API control
                    print("Mission complete.")  # Print mission complete message
                    break  # Exit the loop

                # Move towards the target position
                dgree = math.degrees(math.atan2(target_position[1] - current_position[1], target_position[0] - current_position[0]))  # Calculate the yaw angle to face the target
                client.moveToPositionAsync(target_position[0], target_position[1], z, speed, yaw_mode=airsim.YawMode(False, dgree))  # Move to the target position

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")  # Print interrupt message
        client.armDisarm(False)  # Disarm the drone
        client.reset()  # Reset the simulator
        client.enableApiControl(False)  # Disable API control
    except Exception as e:
        print(f"An error occurred: {e}")  # Print error message
        client.armDisarm(False)  # Disarm the drone
        client.reset()  # Reset the simulator
        client.enableApiControl(False)  # Disable API control

# Main entry point
if __name__ == "__main__":
    client = airsim.MultirotorClient()  # Initialize the AirSim client
    auto_nav_algorithm()  # Start the autonomous navigation algorithm
