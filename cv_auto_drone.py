import airsim
import numpy as np
import cv2
import math
import time

# Function to set segmentation colors for specific objects in the simulation
def set_segmentation_colors():
    print("Setting segmentation colors for objects.")
    client.simSetSegmentationObjectID("[\w]*", 0, True)
    client.simSetSegmentationObjectID('Landscape_1', 0)
    client.simSetSegmentationObjectID('InstancedFoliageActor_0', 1)
    print("Segmentation colors set.")

# Function to calculate the depth planner image
def calc_depth_planner_image():
    print("Capturing depth planner image.")
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])
    response = responses[0]
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    img2d = np.flipud(img2d)
    print("Depth planner image captured.")
    return img2d

# Function to calculate the depth visualization image
def calc_depth_vis_image():
    print("Capturing depth visualization image.")
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])
    response = responses[0]
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    img2d = np.flipud(img2d)
    print("Depth visualization image captured.")
    return img2d

# Function to calculate the segmentation image
def calc_segmentation_image():
    print("Capturing segmentation image.")
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape((response.height, response.width, 3))
    print("Segmentation image captured.")
    return img_rgb

# Main image processing algorithm
def image_proc_algorithm():
    print("Starting image processing algorithm.")
    depth_planner_image = calc_depth_planner_image()
    depth_vis_image = calc_depth_vis_image()
    depth_vis_image = np.flipud(depth_vis_image)
    segmentation_image = calc_segmentation_image()

    out_ = np.ones(depth_vis_image.shape, dtype=depth_vis_image.dtype)
    
    for i in range(depth_vis_image.shape[0]):
        for j in range(depth_vis_image.shape[1]):
            if depth_vis_image[i][j] < 0.2:
                if np.array_equal(segmentation_image[i][j], np.array([42, 174, 203])):
                    out_[i][j] = 0.0
                else:
                    out_[i][j] = 1.0
            else:
                out_[i][j] = 1.0

    out_ = np.flipud(out_)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    erode_image = cv2.erode(out_, kernel)

    erode_image = cv2.copyMakeBorder(erode_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    erode_image = np.flipud(erode_image)

    seg = np.array(erode_image).astype(dtype=np.uint8)
    seg = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX)
    main = np.copy(seg)
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)

    seg = cv2.Canny(seg, 0, 255)

    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out_ = cv2.copyMakeBorder(out_, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    depth_planner_image = cv2.copyMakeBorder(depth_planner_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

    depth = float('inf')
    closest_obstacle = None
    
    for i, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        pos1, pos2, pos3, pos4 = box
        area = abs((pos1[0] * pos2[1] - pos1[1] * pos2[0]) + (pos2[0] * pos3[1] - pos2[1] * pos3[0]) +
                   (pos3[0] * pos4[1] - pos3[1] * pos4[0]) + (pos4[0] * pos1[1] - pos4[1] * pos1[0])) / 2
        
        if area > 1000:
            min_x, max_x = (min([pos1[0], pos2[0], pos3[0], pos4[0]]), max([pos1[0], pos2[0], pos3[0], pos4[0]]))
            min_y, max_y = (min([pos1[1], pos2[1], pos3[1], pos4[1]]), max([pos1[1], pos2[1], pos3[1], pos4[1]]))
            
            if max_x >= np.shape(out_)[1]:
                max_x = np.shape(out_)[1]
            if max_y >= np.shape(out_)[0]:
                max_y = np.shape(out_)[0]
            
            obstacle_depth = float('inf')
            for k in range(min_y, max_y):
                for j in range(min_x, max_x):
                    if out_[k][j] == 0.0:
                        if depth_planner_image[k, j] < obstacle_depth:
                            obstacle_depth = depth_planner_image[k, j]
                        break
            
            if obstacle_depth < depth:
                depth = obstacle_depth
                closest_obstacle = box

            cv2.drawContours(main, [box], 0, (0, 0, 255), 2)

    print("Image processing algorithm completed.")
    return depth, closest_obstacle, main, segmentation_image

# Function to check if there is an obstacle in the given direction
def obstacle_in_path(direction, segmentation_image):
    if direction == "left":
        area = segmentation_image[:, :int(segmentation_image.shape[1] * 0.33)]
    elif direction == "right":
        area = segmentation_image[:, int(segmentation_image.shape[1] * 0.66):]
    elif direction == "forward":
        area = segmentation_image[int(segmentation_image.shape[0] * 0.5):, :]
    elif direction == "backward":
        area = segmentation_image[:int(segmentation_image.shape[0] * 0.5), :]
    else:
        return False

    obstacle_pixels = np.sum(area == np.array([42, 174, 203]))
    return obstacle_pixels > (area.size * 0.1)  # If more than 10% of the area is occupied by obstacles

# Main algorithm for escaping and navigating through obstacles
def Escape_algorithm():
    try:
        print("Starting escape algorithm")
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        set_segmentation_colors()
        client.takeoffAsync().join()
        print("Drone took off successfully.")

        z = -20
        duration = 5
        speed = 1

        point = 0

        while True:
            depth, closest_obstacle, _, segmentation_image = image_proc_algorithm()
            print(f"Depth to closest obstacle: {depth}")

            if depth < 5:
                if closest_obstacle is not None:
                    vx, vy, vz, yaw = 0, 0, 0, 0
                    center_x = (closest_obstacle[0][0] + closest_obstacle[2][0]) / 2
                    center_y = (closest_obstacle[0][1] + closest_obstacle[2][1]) / 2
                    height, width = np.shape(closest_obstacle)[0], np.shape(closest_obstacle)[1]

                    if center_x < 0.33 * width:  # Obstacle is on the left
                        if not obstacle_in_path("left", segmentation_image):
                            vy = speed
                            yaw = 90
                            print("Choosing to move left.")
                        else:
                            print("Left blocked, trying to move right.")
                            vy = -speed
                            yaw = -90
                    elif center_x > 0.66 * width:  # Obstacle is on the right
                        if not obstacle_in_path("right", segmentation_image):
                            vy = -speed
                            yaw = -90
                            print("Choosing to move right.")
                        else:
                            print("Right blocked, trying to move left.")
                            vy = speed
                            yaw = 90
                    else:  # Obstacle is in front
                        if not obstacle_in_path("forward", segmentation_image):
                            if center_y < 0.5 * height:  # Obstacle is above
                                vz = -speed
                                print("Choosing to move down.")
                            else:  # Obstacle is below
                                vz = speed
                                print("Choosing to move up.")
                        else:
                            print("Front blocked, trying to move backward.")
                            vx = -speed

                    print(f"Moving by velocity vx={vx}, vy={vy}, vz={vz}, yaw={yaw}")
                    client.moveByVelocityZAsync(vx, vy, z + vz, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, yaw))
                    time.sleep(1)
            else:
                x = client.getMultirotorState().kinematics_estimated.position.x_val
                y = client.getMultirotorState().kinematics_estimated.position.y_val
                z = client.getMultirotorState().kinematics_estimated.position.z_val
                print(f"Current position: x={x}, y={y}, z={z}")

                if -108 < float(x) < -102 and -83 < float(y) < -77:
                    point = 1
                elif -12 < float(x) < -6 and -110 < float(y) < -100:
                    point = 2
                elif 99 > float(x) > 94 and -107 < float(y) < -100:
                    client.landAsync().join()
                    client.armDisarm(False)
                    client.reset()
                    client.enableApiControl(False)
                    print("Mission complete.")
                    break

                if point == 0:
                    print("Heading to point 1")
                    x1 = -105.86600494384766
                    y1 = -79.34103393554688
                    degree = 180 + math.degrees(math.atan2(y - y1, x - x1))
                    client.moveToPositionAsync(x1, y1, z, speed, yaw_mode=airsim.YawMode(False, degree))
                elif point == 1:
                    print("Heading to point 2")
                    x1 = -8.556684494018555
                    y1 = -105.05833435058594
                    degree = math.degrees(math.atan2(y - y1, x - x1))
                    client.moveToPositionAsync(x1, y1, z, speed, yaw_mode=airsim.YawMode(False, degree))
                elif point == 2:
                    print("Heading to point 3")
                    x1 = 95.15351867675781
                    y1 = -104.22108459472656
                    degree = math.degrees(math.atan2(y - y1, x - x1))
                    client.moveToPositionAsync(x1, y1, z, speed, yaw_mode=airsim.YawMode(False, degree))

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        client.armDisarm(False)
        client.reset()
        client.enableApiControl(False)
    except Exception as e:
        print(f"An error occurred: {e}")
        client.armDisarm(False)
        client.reset()
        client.enableApiControl(False)

# Main entry point
if __name__ == "__main__":
    client = airsim.MultirotorClient()
    Escape_algorithm()
