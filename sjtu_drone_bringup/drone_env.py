import gym.spaces
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import Empty as EmptyDrone
from std_msgs.msg import Empty as EmptyMsg

import matplotlib.pyplot as plt



from cv_bridge import CvBridge
import gym
import cv2
import numpy as np
import time
import os
import math

from spawn_entities import SpawnEntityClient
from delete_entities import DeleteEntityClient
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ContactsState


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        rclpy.init(args=None)
        self.node = Node('drone_env')

        print('**********HELLO FROM MY ENV************')
        self.resetting = True
        self.done = False
        self.width = 160
        self.height = 90
        self.map_height = 10
        self.map_width = 10
        self.channel = 3
        self.visited_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)
        
        self.bridge = CvBridge()

        self.total_reward = 0
        
        self.green_lower = (40, 50, 50)  # Lower bounds for green in HSV
        self.green_upper = (70, 255, 255)  # Upper bounds for green in HSV

        self.collected_targets = []
        self.image = []
        self.detection_reward = 0
        self.current_position = None
        self.last_position = None
        self.start_time = None
        self.prev_frame = None 
        self.image_save_path = "saved_images"  # Directory to save images
        os.makedirs(self.image_save_path, exist_ok=True)
        


        self.delete_entity_client = DeleteEntityClient()
        self.spawn_entity_client = SpawnEntityClient()
     
        self.num_images = 10
        self.images = []

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float16)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.num_images, self.height,self.width, self.channel), dtype=np.uint8)
        # self.observation_space = gym.spaces.Dict({
        #     'image' : gym.spaces.Box(low=0, high=255, shape=(self.height,self.width, self.channel), dtype=np.uint8),
        #     'map' : gym.spaces.Box(low=0, high=1, shape=(self.map_height,self.map_width), dtype=np.uint8),
        # })

        self.image_sub = self.node.create_subscription(Image, '/simple_drone/front/image_raw', self.camera_callback, 10)
        self.current_pose_sub = self.node.create_subscription(Odometry, '/simple_drone/odom', self.position_callback, 10)
        self.collision_sub = self.node.create_subscription(ContactsState, '/simple_drone/bumper_states', self.collision_callback, 10)
       
        self.speed_motors_pub = self.node.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/takeoff', 10)
        self.land_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/land', 10)

        self.reset_client = self.node.create_client(Empty, '/reset_world')
        self.spawn_client = self.node.create_client(SpawnEntity, '/spawn_entity')
        

    def generate_far_point(self, target, min_distance):
        a, b = target
        
        while True:
            # Generate random point [c, d]
            c = 0.0
            d = 0.0
            z_a = np.random.uniform(-1,1)
            
            # Calculate Euclidean distance
            # distance = math.sqrt((c - a)**2 + (d - b)**2)
            
            # # Check if the distance is at least min_distance
            # if distance >= min_distance:
            return c, d, z_a
       
    def reset_simulation(self):
        print("reset called")
        
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            self.node.get_logger().info('Simulation reset successful!')
        else:
            self.node.get_logger().error('Failed to reset simulation')
        # time.sleep(1)    
         
    def position_callback(self, msg:Odometry):
        self._gt_pose = msg

        position = msg.pose.pose.position
        self.current_position = np.array([position.x, position.y, position.z])


        if self.last_position is None:
            self.last_position = self.current_position
            self.start_time = time.time()
    
        else:
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time > 15.0:
                print("Episode time exceeded")
                self.done = True 


    def camera_callback(self, image_msg):

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        b_channel, g_channel, r_channel = cv2.split(cv_image) 
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
       
        # Detect green color
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        b_channel = gray_image
        r_channel = gray_image

        # Merge the modified channels back into a single BGR image
        modified_image = cv2.merge([b_channel, g_channel, r_channel])

          
        # Initialize variables to track the largest contour
        largest_contour = None
        max_area = 0

        # Loop through all contours to find the largest one
        for cnt in green_contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_contour = cnt

        # Create a mask for the largest contour
        largest_contour_mask = np.zeros_like(green_mask)
        # if largest_contour is not None:
        #     cv_image_with_largest_contour = cv_image.copy()
        #     cv2.drawContours(modified_image, [cnt], 0, (0, 255, 0), 2) 
            # cv2.drawContours(cv_image_with_largest_contour, [largest_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

            # Optional: Clear other green balls from the frame by applying the largest contour mask
            # cv_image = cv2.bitwise_and(cv_image, cv_image, mask=largest_contour_mask)

        
        # Count green pixels within the largest contour
        closest_ball_green_pixels = cv2.countNonZero(green_mask)
        self.num_green_pixels = closest_ball_green_pixels

        # green_image = np.expand_dims(g_channel, axis=-1)
        self.green_image = modified_image
        # self.save_image.append(modified_image)

        # Resize the image
        scale_percent = 400  # Percentage of original size
        width = int(modified_image.shape[1] * scale_percent / 100)
        height = int(modified_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(modified_image, dim, interpolation=cv2.INTER_AREA)


        # Display the resized image
        cv2.imshow('Object Detection', resized_image)
        cv2.waitKey(1)


    def moveDrone(self, x: float, y: float,  z_a: float):
        twist_msg = Twist()
        twist_msg.linear.x = x
        twist_msg.linear.y = y
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = z_a
        self.speed_motors_pub.publish(twist_msg)
        time.sleep(1)
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0
        self.speed_motors_pub.publish(twist_msg)
        return True


    def object_name_finder(self):
        print(f"# Green pxl: {self.num_green_pixels}")
        object_name = None
        current_position_xy = self.current_position[:2] 

        distances = {}
        self.dist_array = []
   
        reward = 0
        sum_reward = 0
        for name, self.pos in self.targets.items():
            if name not in self.collected_targets: # Check if target is not collected
                distance = np.linalg.norm(current_position_xy - self.pos[:2])
                self.dist_array.append(distance)
        
                distances[name] = distance


        
        if distances:
            min_target = min(distances, key=distances.get)
            min_distance = distances[min_target]
            self.min_distance = min_distance
            # print(self.min_distance)

            # if min_distance < 0.7:
            if self.num_green_pixels > 100:     
                object_name = f"{min_target}"
                status = self.delete_entity_client.send_request(object_name)
                print("******ALL TARGETS ARE COLLECTED******")
                self.done = True
                self.num_green_pixels = 0
                
        
                # if len(self.dist_array) == 0:  
                   
                
                if object_name in self.targets:
                    del self.targets[object_name]

                if status:
                    self.collected_target_points += 1

                


        # if len(self.dist_array) == 3:
        #     reward = -3 + 2/(1+np.exp(0.5*self.dist_array[0]/(100*self.num_green_pixels+1)))
        #     +2/(1+np.exp(0.5*self.dist_array[1]/(100*self.num_green_pixels+1))) 
        #     +2/(1+np.exp(0.5*self.dist_array[2]/(100*self.num_green_pixels+1)))  

        # elif len(self.dist_array) == 2:
        #     reward = -2 + 2/(1+np.exp(1/(100*self.num_green_pixels+1)))
        #     reward = -2 + 2/(1+np.exp(0.5*self.dist_array[0]/(100*self.num_green_pixels+1)))
        #     +2/(1+np.exp(0.5*self.dist_array[1]/(100*self.num_green_pixels+1))) 

          
        # print(f"dist: {self.dist_array}")
                   

        # reward = -1 + 2/(1+np.exp(5*self.min_distance)/(self.num_green_pixels+1))  
        # reward = -1 + 2 / (1 + np.exp(self.min_distance / (self.num_green_pixels+1)))          
        # print(f"green pix: {self.num_green_pixels}")
       
        return object_name

    def collision_callback(self, msg):
        if self.resetting:
            return
        for state in msg.states:
            # Extract the collision names
            self.collision1_name = state.collision1_name
            self.collision2_name = state.collision2_name

            # Check if either collision name contains "Wall"
            if 'Wall' in self.collision1_name or 'Wall' in self.collision2_name:
                print("Collision with a wall detected!")
                self.total_reward -= 1
                self.done = True
                    

    def position_to_grid(self, pos, x_min, x_max, y_min, y_max, grid_size=10):
        x, y = pos
        
        # Calculate grid cell size
        cell_width = (x_max - x_min) / grid_size
        cell_height = (y_max - y_min) / grid_size
        
        # Normalize position
        x_normalized = x - x_min
        y_normalized = y - y_min
        
        # Calculate grid coordinates
        grid_x = int(x_normalized / cell_width)
        grid_y = int(y_normalized / cell_height)
        
        # Ensure grid coordinates are within the grid bounds
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        
        return grid_x, grid_y

    def take_action(self, action):
        self.object_name_finder()

        # linear_x = (action[0]+1)/2
        # angular_z = action[1]
       
        linear_x = (action[0]+1)/2
        angular_z = action[1]
        # print(f"linear {linear_x}, angular {angular_z}")
        
        vel_cmd = Twist()
        vel_cmd.linear.x = float(5*linear_x)
        vel_cmd.angular.z = float(5*angular_z)
        
        self.speed_motors_pub.publish(vel_cmd)

        time.sleep(0.5)

        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        
        self.speed_motors_pub.publish(vel_cmd)


        
       
    def step(self, action):
        self.take_action(action)

        if len(self.dist_array) == 1:
            reward = -1 + 2/(1+np.exp(self.dist_array[0]/(self.num_green_pixels+1)))
            self.total_reward = reward
            print(f"Reward: {self.total_reward}")    
 
        self.observation = self.green_image

        info = {}
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        return self.observation, self.total_reward, self.done, info

    def reset(self):
        # print("Reset function called")
        self.resetting = True
        self.action_count = 0
        
        self.last_position = None
        self.start_time = 0.0
        self.elapsed_time = 0.0
       
        self.green_pixels = 0
        self.collected_target_points = 0
        self.current_position = [0,0,0]
        self.total_reward = 0
        self.coll = False
        self.cnt = 0
        self.collision1_name = ''
        self.collision2_name = ''

        self.min_distance = np.array([100])
        # self.visited_map.fill(0)

        grey_image = np.zeros((self.height, self.width, self.channel), dtype=np.uint8)
        self.grey_image = np.array(grey_image)
        green_image = np.zeros((self.height, self.width, self.channel), dtype=np.uint8)
        self.green_image = np.array(green_image)
        # pos = self.current_position[:2]
        # self.observation = pos
        # self.observation = {'image': self.grey_image, 'map': self.visited_map}
        self.observation = self.green_image

        
        self.land_publisher.publish(EmptyMsg())
        self.node.get_logger().info("Land command sent")


        self.delete_entity_client.send_request('green_ball_0')
        # self.delete_entity_client.send_request('green_ball_1')
        # self.delete_entity_client.send_request('green_ball_2')
        # self.targets = None
        

        
        self.reset_simulation()
        self.node.get_logger().info("Reset simulation command sent")
        time.sleep(1)
        self.targets = self.spawn_entity_client.send_request('green_ball', 1)
        time.sleep(1)

        min_distance = 2
        pos = self.targets['green_ball_0']
        target = [pos[0],pos[1]]
        x,y,z_a = self.generate_far_point(target, min_distance)
                
        

        if self.targets is not None:
            self.takeoff_publisher.publish(EmptyMsg())
            self.node.get_logger().info("Takeoff command sent")
            # time.sleep(2)
            self.moveDrone(x,y,z_a)
            self.node.get_logger().info("Drone moved to initial pos")
            time.sleep(1)
        else:
            self.node.get_logger().error("No targets available, not taking off") 
        
        # print("+++++++++++++++++++++++++++++++++++++")
        # print(self.targets)
        # print("+++++++++++++++++++++++++++++++++++++")
        

        self.done = False
        self.resetting = False
        return self.observation


    def close(self):
        # Clean up ROS 2 resources
        self.node.destroy_node()
        rclpy.shutdown()
        # self.ros_thread.join()


