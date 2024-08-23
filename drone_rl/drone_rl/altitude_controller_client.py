import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import time

class AltitudeControllerClient(Node):
    def __init__(self, drone_id):
        super().__init__('altitude_controller_client')

        # Initialize variables
        self.current_altitude = 0.0
        self.target_altitude = 10.0
        self.tolerance = 0.05  # Tolerance for altitude

        # Create a publisher for the velocity command
        self.vel_pub = self.create_publisher(Twist, f'/{drone_id}/cmd_vel', 10)

        # Create a subscriber to get the drone's current altitude from the odometry data
        self.odom_sub = self.create_subscription(Odometry, f'/{drone_id}/odom', self.odom_callback, 10)

        # Create a service server to set the target altitude
        self.srv = self.create_service(SetBool, 'set_target_altitude', self.set_target_altitude_callback)

        # Create a client to send altitude commands
        self.cli = self.create_client(SetBool, 'set_target_altitude')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def odom_callback(self, msg):
        # Extract the altitude (z) from the odometry message
        self.current_altitude = msg.pose.pose.position.z

    def set_target_altitude_callback(self, request, response):
        # Update the target altitude based on the service request
        if request.data:
            self.target_altitude = 10.0
        else:
            self.target_altitude = 0.0
        response.success = True
        response.message = f"Target altitude set to {self.target_altitude} meters"
        return response

    def send_request(self, altitude_set):
        # Send a request to set the target altitude
        req = SetBool.Request()
        req.data = altitude_set
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response.success:
            self.get_logger().info(f'Service call successful: {response.message}')
        else:
            self.get_logger().error('Service call failed')

    def move_to_altitude(self):
        vel_cmd = Twist()

        while rclpy.ok():
            # Calculate the altitude error
            altitude_error = self.target_altitude - self.current_altitude

            # If the altitude is within the tolerance range, stop the drone
            if abs(altitude_error) < self.tolerance:
                # self.get_logger().info(f"Reached target altitude: {self.current_altitude} meters")
                break

            # Set the vertical velocity proportional to the altitude error
            vel_cmd.linear.z = 0.5 * altitude_error

            # Publish the velocity command
            self.vel_pub.publish(vel_cmd)

            # Log the current altitude
            # self.get_logger().info(f"Current altitude: {self.current_altitude:.2f} meters, moving to {self.target_altitude} meters")

            # Spin once to process the odometry callback
            rclpy.spin_once(self)

            # Sleep for a short duration to maintain the loop rate
            time.sleep(0.1)

        # Stop the drone's vertical movement
        vel_cmd.linear.z = 0.0
        self.vel_pub.publish(vel_cmd)
        # self.get_logger().info("Stopping vertical movement.")

def main(args=None):
    rclpy.init(args=args)
    altitude_controller_client = AltitudeControllerClient()

    try:
        # Set the target altitude and move to it
        altitude_controller_client.send_request(True)
        altitude_controller_client.move_to_altitude()
    finally:
        altitude_controller_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
