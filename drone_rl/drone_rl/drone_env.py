import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Range
from std_srvs.srv import Empty
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Bool


from cv_bridge import CvBridge
import gym
import cv2
import numpy as np
import time
from collections import deque

from drone_rl.spawn_entities import SpawnEntityClient
from drone_rl.goal_position import GoalClient
from drone_rl.delete_entities import DeleteEntityClient
from drone_rl.altitude_controller_client import AltitudeControllerClient
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ContactsState

import torch as T
import math
from pyquaternion import Quaternion
from visualization_msgs.msg import Marker
from torch.utils.tensorboard import SummaryWriter
# from ddpg import DDPGNode
import tf_transformations



class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        rclpy.init(args=None)
        self.node = Node('drone_env')

        print('********** HELLO FROM MY ENV ************')
        logdir = f"logs/{int(time.time())}/"
        self.writer = SummaryWriter(logdir)
        self.resetting = True
        self.episode_number = 0
        self.success = 0
        self.fail = 0
        self.goal_reached = False
        self.terminated = False # when time limit is reached     
        self.truncated = False
        self.stop_signal = False
        # self.ddpg = DDPGNode()
        
        self.agent_position = np.array([0.0, 0.0, 0.0])
        self.last_position = None
        self.start_time = None
        self.isFlying = False
        self.episode_rewards = []

        self.delete_entity_client = DeleteEntityClient()
        self.spawn_entity_client = SpawnEntityClient()
        self.drone_controller = AltitudeControllerClient('drone1')

        self.goal_client = GoalClient()

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None


        self.width = 640
        self.height = 360

        self.observation_space = gym.spaces.Box(low=0, high=640, shape=(3,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=0, high=640, shape=(7,), dtype=np.float32)

        action_low = np.array([-1, -1, -1], dtype=np.float32)
        action_high = np.array([1, 1, 1], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.stop_signal_sub = self.node.create_subscription(Bool, '/drone1/stop_rl_node', self.stop_signal_callback, 10)

        self.depth_sub = self.node.create_subscription(Image, '/drone1/front_camera/depth/image_raw', self.depth_callback, 10)
        self.rgb_sub = self.node.create_subscription(Image, '/drone1/front_camera/image_raw', self.rgb_callback, 10)
        
        self.current_pose_sub = self.node.create_subscription(Odometry, '/drone1/odom', 
                                                              self.position_callback, 1024)
        
        self.collision_sub = self.node.create_subscription(ContactsState, '/drone1/bumper_states', 
                                                           self.collision_callback, 10)
       

        self.speed_motors_pub = self.node.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.takeoff_publisher = self.node.create_publisher(EmptyMsg, '/drone1/takeoff', 10)
        self.land_publisher = self.node.create_publisher(EmptyMsg, '/drone1/land', 10)

        self.reset_client = self.node.create_client(Empty, '/reset_world')



    def collision_callback(self, msg):
        self.wall = 0
        if self.resetting:
            return
        
        for state in msg.states:
            # Extract the collision names
            self.collision1_name = state.collision1_name
            self.collision2_name = state.collision2_name

            # Check if either collision name contains "Wall"
            # if 'Wall' in self.collision1_name or 'Wall' in self.collision2_name:
            if 'collapsed_industrial' in self.collision1_name or 'collapsed_industrial' in self.collision2_name:
                print("Collision with a wall detected!")
                self.wall = 1
                # self.terminated = True    

    def rgb_callback(self, msg):
        if msg is None:
            return
        self.rgb_frame_ = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = np.nan_to_num(self.rgb_frame_, nan=0, posinf=0)
        # self.rgb_image = copy.deepcopy(self.rgb_frame_)
        
    def depth_callback(self, msg):
        if msg is None:
            return
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

        # # Define blue color range in HSV
        # lower_blue = np.array([100, 150, 0])
        # upper_blue = np.array([140, 255, 255])

        bgr_blue = np.uint8([[[255, 0, 0]]])  # BGR format for blue

        # Convert BGR blue to HSV
        hsv_blue = cv2.cvtColor(bgr_blue, cv2.COLOR_BGR2HSV)
        hue_blue = hsv_blue[0][0][0]  # Extract the hue value

        # Define a narrow range around this hue value
        lower_blue = np.array([hue_blue - 10, 100, 100])
        upper_blue = np.array([hue_blue + 10, 255, 255])

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

        self.depth_image = np.nan_to_num(self.depth_image, nan=np.inf)  # Convert NaNs to inf
        min_depth = np.nanmin(self.depth_image[np.isfinite(self.depth_image)])  # Min depth ignoring inf
        max_depth = np.nanmax(self.depth_image[np.isfinite(self.depth_image)])  # Max depth ignoring inf

        # Clip depth image to avoid overflow
        clipped_depth_image = np.clip(self.depth_image, min_depth, max_depth)

        # Normalize the depth image to the range 0-255
        normalized_depth = np.nan_to_num(clipped_depth_image, nan=0).astype(np.uint8)
        normalized_depth = 255 * (normalized_depth - min_depth) / (max_depth - min_depth + 1e-5)

        _, _, d = self.find_blue_object()
        if d == 0:
            self.closest_distance = np.nanmin(self.depth_image)
            # print(f"Closest distance to an obstacle: {self.closest_distance}")
    
        
        cv2.imshow('Depth Image', normalized_depth)
        cv2.waitKey(1)
        
    def take_action(self, v_yaw, v_x, v_z):
        vel_cmd = Twist()
        vel_cmd.linear.x  = 3.0*v_x
        vel_cmd.linear.z = v_z
        vel_cmd.angular.z = v_yaw
        
        self.speed_motors_pub.publish(vel_cmd)

    def takeOff(self):
        """
        Take off the drone
        :return: True if the command was sent successfully, False if drone is already flying
        """
        if self.isFlying:
            return False
        self.node.get_logger().info("Taking off")
        self.takeoff_publisher.publish(EmptyMsg())
        self.isFlying = True
        return True

    def land(self):
        """
        Land the drone
        :return: True if the command was sent successfully, False if drone is not flying
        """
        if not self.isFlying:
            return False
        self.node.get_logger().info("Landing")
        self.land_publisher.publish(EmptyMsg())
        self.isFlying = False
        return True
       
    def reset_simulation(self):
        print("reset called")
        
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            self.node.get_logger().info('Simulation reset successful!')
        else:
            self.node.get_logger().error('Failed to reset simulation')  
     

    def position_callback(self, msg):
        position = msg.pose.pose.position
        # quaternion = msg.pose.pose.orientation

        # roll, pitch, yaw = self.quaternion_to_euler(quaternion)

        self.agent_position = np.array([position.x, position.y, position.z])
        self.current_altitude = msg.pose.pose.position.z
        
        # self.agent_orientation = np.array([roll, pitch, yaw])

        if self.last_position is None:
            self.last_position = self.agent_position
            self.start_time = time.time()
            
    
        else:
            self.elapsed_time = time.time() - self.start_time

            if self.elapsed_time > 60.0:
                print("TIME EXCEEDED")
                self.truncated = True            

    def calculate_dist_angle(self): 
        x, y, d = self.find_blue_object()
        if d < 0.0:
            d = 0.0
        return np.array([x,y,d])

    def get_observation(self):
        state = self.calculate_dist_angle()
        observation = [state[0], state[1], state[2]]
        # print(f"Observation: {observation}")
        
        return observation   

        
    def step(self, action):
        if self.stop_signal:
            self.node.get_logger().info("Stopping the RL node due to stop signal.")
            self.close()
            return None, None, True, True, {}


        self.take_action(float(action[0]), float(action[1]), float(action[2]))
        x,y, distance = self.calculate_dist_angle() #TODO

        self.distance = distance
        
        reward = 0.0
        penalty = 0.0
        penalty_distance = 0.0
        reward_collision = 0.0
        reward_proximity = 0.0

        agent_position = np.array(self.agent_position)
        goal_position = np.array(self.goal_position)
        distance_to_goal = math.dist(goal_position, agent_position)
        # print(f'x pos: {self.agent_position[0]}, y pos: {self.agent_position[1]}')

        if distance_to_goal < 3.0: # needs to see the goal but doesnt really necessary to approach it
            print("******REACHED THE GOAL POSITION******") 
            self.success += 1 
            self.terminated = True


        if self.distance == 0.0:
            penalty = -10.0

        if self.agent_position[2] < 1.0:
            reward_collision = -10.0   
     
        if abs(self.agent_position[0]) > 10.0 or abs(self.agent_position[1]) > 10.0: 
            penalty_distance = -10.0
            print("Out of the region")   

        if self.closest_distance < 2.0 and self.wall == 0:
            reward_collision = -10.0
        if self.wall == 1:
            print("Collision") 
            self.fail += 1   
            self.terminated = True         
  
        
        reward = -1*distance_to_goal + penalty + reward_collision + penalty_distance
        self.episode_rewards.append(reward)

        self.update_status()
        
        # print(f"reward: {reward}") # TODO

        observation = self.get_observation()
        if self.episode_number < 11 and self.elapsed_time > 5.0:
            self.success_plot()

        
        rclpy.spin_once(self.node, timeout_sec= 0.1)

        return observation, reward, self.terminated, self.truncated, {}
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.episode_number += 1
        # self.episode_rewards.append(0)
        print(f"Reset function called: {self.episode_number}") # TODO
        self.resetting = True
        self.terminated = False
        self.truncated = False
        
        self.last_position = None
        self.start_time = 0.0
        self.elapsed_time = 0.0

        self.collision1_name = ''
        self.collision2_name = ''

        self._on_episode_end()
        self.distance = 10.0
        self.agent_position = np.array([0.0, 0.0, 0.0])
        self.wall = 0
        self.closest_distance = 10.0

        self.land()
        
        self.reset_simulation()
        time.sleep(2) #needs a bit time to take off 

        self.takeOff()

        self.set_and_move_to_altitude('drone1',True)

        data = self.goal_client.send_request('goal_circle_blue', -2.0, 7.0, 11.0)
        coordinates = data['goal_circle_blue']
        self.goal_position = np.array([coordinates[0], coordinates[1], coordinates[2]])

        time.sleep(1)

        observation = np.array([0, 0, 0.0])
        info = {}

        self.resetting = False
        return observation, info

    def set_and_move_to_altitude(self, drone_id, altitude_set):
        # Select the appropriate controller based on drone_id
        if drone_id == 'drone1':
            controller = self.drone_controller
        else:
            raise ValueError("Unknown drone_id")

        # Send a request to set the target altitude
        controller.send_request(altitude_set)
        
        # Move to the target altitude
        controller.move_to_altitude()

    def close(self):
        # Clean up ROS 2 resources
        self.node.destroy_node()
        rclpy.shutdown()

    def get_avg_reward(self):
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            self.episode_rewards = []  # Reset for next calculation
            return avg_reward
        else:
            return 0

    def _on_episode_end(self) -> bool:
        avg_reward = self.get_avg_reward()
        # print(f"AVG REWARD: {avg_reward}") # TODO
        self.writer.add_scalar('Average Reward', avg_reward, self.episode_number)
        return True     

    def success_plot(self):
        self.writer.add_scalar('Success Rate', self.success, self.episode_number)