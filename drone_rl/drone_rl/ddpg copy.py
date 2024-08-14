import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np
import copy

class PID:
    def __init__(self, Kp, Kd, Ki, output_limits=(None, None), integral_limits=(None, None), anti_windup_gain=1.0):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.prev_error = 0
        self.integral = 0
        self.output_min, self.output_max = output_limits
        self.integral_min, self.integral_max = integral_limits
        self.anti_windup_gain = anti_windup_gain

    def compute(self, error, delta_time=0.01):
        self.integral += error * delta_time
        
        # Clamp the integral term
        if self.integral_min is not None:
            self.integral = max(self.integral_min, self.integral)
        if self.integral_max is not None:
            self.integral = min(self.integral_max, self.integral)
        
        derivative = (error - self.prev_error) / delta_time
        output = self.Kp * error + self.Kd * derivative + self.Ki * self.integral
        self.prev_error = error

        # Clamp the output
        if self.output_min is not None and output < self.output_min:
            output = self.output_min
            self.integral -= self.anti_windup_gain * (output - self.output_min) * delta_time
        elif self.output_max is not None and output > self.output_max:
            output = self.output_max
            self.integral -= self.anti_windup_gain * (output - self.output_max) * delta_time

        return output


class DDPGNode(Node):
    def __init__(self):
        super().__init__('ddpg_node')

        self.bridge = CvBridge()
        # self.depth_sub = self.create_subscription(Image, '/depth_info', self.depth_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/simple_drone/front_camera/depth/image_raw', self.depth_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/simple_drone/front_camera/image_raw', self.rgb_callback, 10)
        self.speed_motors_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)

        self.input_dims = (1, 640, 360)

        # Initialize PID gains
        self.kP_dist = 4.0  # Proportional gain
        self.kI_dist = 0.0  # Integral gain
        self.kD_dist = 0.0  # Derivative gain

        # Initialize PID gains
        self.kP_orient = 1.0  # Proportional gain
        self.kI_orient = 0.0  # Integral gain
        self.kD_orient = 0.0  # Derivative gain

        self.fov_horizontal = 1.047
        

        self.rgb_image = None
        self.depth_image = None

        # reference
        self.x_img = 319
        self.y_img = 179
        self.d = 0.5

        self.pid_distance = PID(self.kP_dist, self.kI_dist, self.kD_dist)
        self.pid_orient = PID(self.kP_orient, self.kI_orient, self.kD_orient)

    def interpolate(self, x, in_min=0, in_max=640, out_min=-np.pi/2, out_max=np.pi/2):
        # Calculate the interpolated value
        value = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return value

    def rgb_callback(self, msg):
        self.rgb_frame_ = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = np.nan_to_num(self.rgb_frame_, nan=0, posinf=0)
        # self.rgb_image = copy.deepcopy(self.rgb_frame_)
        
         

    def depth_callback(self, msg):
        self.depth_frame_ = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        # self.depth_image = copy.deepcopy(self.depth_frame_)
        self.depth_image = np.nan_to_num(self.depth_frame_, nan=0, posinf=0)
        self.process_frames()

    def find_blue_object(self):
        if self.rgb_image is None:
            print("Error: RGB frame is not initialized or loaded.")
            return 0, 0, 0.0
        
        # Convert RGB image to HSV
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)

        # Define blue color range in HSV
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0, 0, 0.0
        
        # Assume the largest contour corresponds to the blue object
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return 0, 0, 0.0
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # Ensure coordinates are within image bounds
        height, width = self.depth_image.shape
        center_x = max(0, min(center_x, width - 1))
        center_y = max(0, min(center_y, height - 1))

        # Extract the depth value at the centroid
        depth_value = self.depth_image[center_y, center_x]

        return center_x, center_y, depth_value
    
    def process_frames(self):
        # Convert the reference
        x_ref = self.interpolate(self.x_img)
        y_ref = self.interpolate(self.y_img)

        x, y, d = self.find_blue_object()
        # self.get_logger().info(f'x,y,d: {x, y, d}')
        x_act = self.interpolate(x)
        y_act = self.interpolate(y)
        # Calculate min and max depth values ignoring NaNs
        min_depth = np.nanmin(self.depth_image)
        max_depth = np.nanmax(self.depth_image)

        # # Normalize the depth image to the range 0-255
        normalized_depth = 255 * (self.depth_image - min_depth) / (max_depth - min_depth)
        normalized_depth = np.nan_to_num(normalized_depth, nan=0).astype(np.uint8)

        error_dist = d-self.d
        error_orient = x_ref -x_act

        vx = self.pid_distance.compute(error_dist)
        v_psi = self.pid_distance.compute(error_orient)

        

        # print(f'error_dist: {error_dist}, vx: {vx}')
        # print(f'error_orient: {error_orient}, v_psi: {v_psi}, error_dist: {error_dist}, vx: {vx}')
        
       


        # # Convert PID outputs to velocities
        linear_x_vel = vx
        angular_z_vel = v_psi
        # self.take_action(angular_z_vel, linear_x_vel)

        # cv2.imshow('Depth Image', self.depth_image)
        # cv2.waitKey(1)
        return np.array([x_ref, y_ref, self.d, x_act, y_act, d])


    def take_action(self, v_yaw, v_x):
        vel_cmd = Twist()
        vel_cmd.linear.x  = v_x
        vel_cmd.angular.z = v_yaw
        
        self.speed_motors_pub.publish(vel_cmd)


def main(args=None):
    rclpy.init()
    ddpg = DDPGNode()
    try:
        while rclpy.ok():
            rclpy.spin(ddpg)
            
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    

if __name__ == '__main__':
    main()
