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

        self.drone1_position_sub = self.create_subscription(Odometry, '/drone1/odom', self.drone1_position_callback, 1024)
        self.drone2_position_sub = self.create_subscription(Odometry, '/drone2/odom', self.drone2_position_callback, 1024)

        self.drone1_cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 1024)
        self.drone2_cmd_pub = self.create_publisher(Twist, '/drone2/cmd_vel', 1024)


        self.drone1_position = np.array([0.0, 0.0, 0.0])
        self.drone2_position = np.array([1.0, -1.0, 0.0])

        self.drone1_detected = False
        self.drone2_detected = False

        # Create timers to check distances and target detection
        self.create_timer(0.5, self.check_distances)
        # self.create_timer(0.1, self.check_target_detection)

    def drone1_position_callback(self, msg):
        position = msg.pose.pose.position
        self.drone1_position = np.array([position.x, position.y, position.z])
        

    def drone2_position_callback(self, msg):
        position = msg.pose.pose.position
        self.drone2_position = np.array([position.x, position.y, position.z])

    def check_distances(self):
        # Calculate distance between drones
        distance = np.linalg.norm(self.drone1_position - self.drone2_position)

        drone1_cmd = Twist()
        drone2_cmd = Twist()

        if distance > 10.0:
            self.get_logger().info("Drones are too far apart! Correcting positions.")
            drone1_cmd.linear.x = 0.5 * (self.drone2_position[0] - self.drone1_position[0])
            drone1_cmd.linear.y = 0.5 * (self.drone2_position[1] - self.drone1_position[1])
            drone2_cmd.linear.x = -0.5 * (self.drone2_position[0] - self.drone1_position[0])
            drone2_cmd.linear.y = -0.5 * (self.drone2_position[1] - self.drone1_position[1])

        elif distance < 1.0:
            
            self.get_logger().info("Drones are too close! Correcting positions.")

            drone1_cmd.linear.x = -0.5 * (self.drone2_position[0] - self.drone1_position[0])
            drone1_cmd.linear.y = -0.5 * (self.drone2_position[1] - self.drone1_position[1])
            drone2_cmd.linear.x = 0.5 * (self.drone2_position[0] - self.drone1_position[0])
            drone2_cmd.linear.y = 0.5 * (self.drone2_position[1] - self.drone1_position[1])
            
        self.drone1_cmd_pub.publish(drone1_cmd)
        self.drone2_cmd_pub.publish(drone2_cmd)

def main(args=None):
    rclpy.init(args=args)
    drone_manager = DroneManager()
    rclpy.spin(drone_manager)

    drone_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
