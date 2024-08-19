import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import numpy as np
import time
import math

class DroneManager(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # Define publishers and subscribers
        self.drone1_target_pub = self.create_publisher(Bool, '/drone1/target_detected', 10)
        self.drone2_target_pub = self.create_publisher(Bool, '/drone2/target_detected', 10)
        self.drone1_target_sub = self.create_subscription(Bool, '/drone1/target_detected', self.drone1_target_callback, 10)
        self.drone2_target_sub = self.create_subscription(Bool, '/drone2/target_detected', self.drone2_target_callback, 10)

        self.drone1_stop_rl_pub = self.create_publisher(Bool, '/drone1/stop_rl_node', 10)
        self.drone2_stop_rl_pub = self.create_publisher(Bool, '/drone2/stop_rl_node', 10)

        self.drone1_position_sub = self.create_subscription(Odometry, '/drone1/odom', self.drone1_position_callback, 1024)
        self.drone2_position_sub = self.create_subscription(Odometry, '/drone2/odom', self.drone2_position_callback, 1024)

        self.drone1_cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.drone2_cmd_pub = self.create_publisher(Twist, '/drone2/cmd_vel', 10)


        self.drone1_position = None
        self.drone2_position = None

        self.drone1_detected = False
        self.drone2_detected = False

        # Create timers to check distances and target detection
        self.create_timer(0.5, self.check_distances)
        self.create_timer(0.1, self.check_target_detection)

    def drone1_position_callback(self, msg):
        position = msg.pose.pose.position
        self.drone1_position = np.array([position.x, position.y, position.z])
        

    def drone2_position_callback(self, msg):
        position = msg.pose.pose.position
        self.drone2_position = np.array([position.x, position.y, position.z])

    def drone1_target_callback(self, msg):
        if msg.data:
            self.drone1_detected = True
            self.get_logger().info("Drone 1 detected the target, resetting Drone 2.")
            stop_msg = Bool()
            stop_msg.data = True
            self.drone2_stop_rl_pub.publish(stop_msg)

    def drone2_target_callback(self, msg):
        if msg.data:
            self.drone2_detected = True
            self.get_logger().info("Drone 2 detected the target, resetting Drone 1.")
            stop_msg = Bool()
            stop_msg.data = True
            self.drone1_stop_rl_pub.publish(stop_msg)

    def check_distances(self):
        # Calculate distance between drones
        distance = np.linalg.norm(self.drone1_position - self.drone2_position)

        drone1_cmd = Twist()
        drone2_cmd = Twist()

        if distance > 10.0:
            self.get_logger().info("Drones are too far apart! Correcting positions.")

            drone1_cmd.linear.x = 0.5 * (self.drone2_position.x - self.drone1_position.x)
            drone1_cmd.linear.y = 0.5 * (self.drone2_position.y - self.drone1_position.y)
            drone2_cmd.linear.x = -0.5 * (self.drone2_position.x - self.drone1_position.x)
            drone2_cmd.linear.y = -0.5 * (self.drone2_position.y - self.drone1_position.y)

        elif distance < 1.0:
            
            self.get_logger().info("Drones are too close! Correcting positions.")

            drone1_cmd.linear.x = -0.5 * (self.drone2_position.x - self.drone1_position.x)
            drone1_cmd.linear.y = -0.5 * (self.drone2_position.y - self.drone1_position.y)
            drone2_cmd.linear.x = 0.5 * (self.drone2_position.x - self.drone1_position.x)
            drone2_cmd.linear.y = 0.5 * (self.drone2_position.y - self.drone1_position.y)
            
        self.drone1_cmd_pub.publish(drone1_cmd)
        self.drone2_cmd_pub.publish(drone2_cmd)

    def check_target_detection(self):
        if self.drone1_detected:
            self.drone2_detected = False
            self.reset_drone2()

        elif self.drone2_detected:
            self.drone1_detected = False
            self.reset_drone1()

    def reset_drone1(self):
        self.get_logger().info("Resetting Drone 1")
        # Add code to reset Drone 1

    def reset_drone2(self):
        self.get_logger().info("Resetting Drone 2")
        # Add code to reset Drone 2

def main(args=None):
    rclpy.init(args=args)
    drone_manager = DroneManager()
    rclpy.spin(drone_manager)

    drone_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
