import gym.spaces
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Range
from std_srvs.srv import Empty
from std_msgs.msg import Empty as EmptyMsg


from cv_bridge import CvBridge
import gym
import cv2
import numpy as np
import time
from collections import deque

from spawn_entities import SpawnEntityClient
from goal_position import GoalClient
from delete_entities import DeleteEntityClient
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ContactsState

import torch as T
import math
from pyquaternion import Quaternion
from visualization_msgs.msg import Marker
from torch.utils.tensorboard import SummaryWriter
from ddpg import DDPGNode




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
        self.terminated = False # when time limit is reached
        self.ddpg = DDPGNode()
        
        self.agent_position = np.array([0.0, 0.0, 0.0])
        self.last_position = None
        self.start_time = None
        self.isFlying = False
        self.episode_rewards = []

        self.delete_entity_client = DeleteEntityClient()
        self.spawn_entity_client = SpawnEntityClient()
        self.goal_client = GoalClient()

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None


        self.width = 640
        self.height = 360

        self.observation_space = gym.spaces.Box(low=0, high=640, shape=(3,), dtype=np.float32)

        action_low = np.array([-1,-1], dtype=np.float32)
        action_high = np.array([1, 1], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.depth_sub = self.node.create_subscription(Image, '/simple_drone/front_camera/depth/image_raw', self.depth_callback, 10)
        self.rgb_sub = self.node.create_subscription(Image, '/simple_drone/front_camera/image_raw', self.rgb_callback, 10)
        
        self.current_pose_sub = self.node.create_subscription(Odometry, '/simple_drone/odom', 
                                                              self.position_callback, 1024)

        self.speed_motors_pub = self.node.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/takeoff', 10)
        self.land_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/land', 10)



        self.reset_client = self.node.create_client(Empty, '/reset_world')

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

        min_depth = np.nanmin(self.depth_image)
        max_depth = np.nanmax(self.depth_image)

        # # Normalize the depth image to the range 0-255
        normalized_depth = 255 * (self.depth_image - min_depth) / (max_depth - min_depth)
        normalized_depth = np.nan_to_num(normalized_depth, nan=0).astype(np.uint8)

        cv2.imshow('Depth Image', normalized_depth)
        cv2.waitKey(1)
        
    def take_action(self, v_yaw, v_x):
        vel_cmd = Twist()
        vel_cmd.linear.x  = v_x
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

        self.agent_position = np.array([position.x, position.y, position.z])

        if self.last_position is None:
            self.last_position = self.agent_position
            self.start_time = time.time()
            
    
        else:
            self.elapsed_time = time.time() - self.start_time

            if self.elapsed_time > 20.0: #TODO:change it back to 20
                print("TIME EXCEEDED")
                self.truncated = True            

    def calculate_dist_angle(self): 
        x, y, d = self.find_blue_object()
        return np.array([x,y,d])

    def get_observation(self):
        state = self.calculate_dist_angle()
        observation = [state[0], state[1], state[2]]
        print(f"Observation: {observation}")
        
        return observation
    
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

        
    def step(self, action):
        self.take_action(float(action[0]), float(action[1]))
        x,y, distance = self.calculate_dist_angle() #TODO
        self.distance = distance
        
        reward_complete = 0.0
        penalty = 0.0
        agent_position = np.array(self.agent_position[:2])
        goal_position = np.array(self.goal_position[:2])
        distance_to_goal = math.dist(goal_position, agent_position)


        if distance_to_goal < 1.0:
            print("******REACHED THE GOAL POSITION******") 
            self.terminated = True
        if self.distance == 0.0:
            penalty = -10.0

        if x != 0:
            x_reward = -1e-2*(x-319)
        else:
            x_reward = -1.0    
    
        reward = -1*distance_to_goal + penalty + x_reward
        self.episode_rewards.append(reward)
        
        print(f"reward: {reward}") # TODO

        observation = self.get_observation()

        
        rclpy.spin_once(self.node, timeout_sec= 1.0)

        return observation, reward, self.terminated, self.truncated, {}
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.episode_number += 1
        self.episode_rewards.append(0)
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

        self.land()
        
        self.reset_simulation()
        # self.delete_entity_client.send_request('goal_circle')
        time.sleep(2) #needs a bit time to take off 

        self.takeOff()

        data = self.goal_client.send_request('goal_circle')
        coordinates = data['goal_circle']
        self.goal_position = np.array([coordinates[0], coordinates[1], coordinates[2]])

        time.sleep(1)

        observation = np.array([0, 0, 0.0])
        info = {}

        self.resetting = False
        return observation, info
    
    def close(self):
        # Clean up ROS 2 resources
        self.node.destroy_node()
        rclpy.shutdown()

