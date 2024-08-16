import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

class TakeoffPublisher(Node):

    def __init__(self):
        super().__init__('takeoff_publisher')
        
        # Create publishers for both drone topics
        self.takeoff_publisher_drone_1 = self.create_publisher(Empty, '/drone_1/takeoff', 10)
        self.takeoff_publisher_drone_2 = self.create_publisher(Empty, '/drone_2/takeoff', 10)

        # Timer to periodically publish messages
        self.timer = self.create_timer(1.0, self.timer_callback)  # Publish every second

    def timer_callback(self):
        # Create and publish an empty message for drone_1
        msg = Empty()
        self.takeoff_publisher_drone_1.publish(msg)
        self.get_logger().info('Publishing takeoff message to /drone_1/takeoff')

        # Create and publish an empty message for drone_2
        self.takeoff_publisher_drone_2.publish(msg)
        self.get_logger().info('Publishing takeoff message to /drone_2/takeoff')

def main(args=None):
    rclpy.init(args=args)
    node = TakeoffPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
