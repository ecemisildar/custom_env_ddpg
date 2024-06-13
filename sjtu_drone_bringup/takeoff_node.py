import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

class TakeoffPublisher(Node):

    def __init__(self):
        super().__init__('takeoff_node')
        self.publisher_ = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Empty()
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing Empty message to /simple_drone/takeoff')
        self.i += 1
        if self.i >= 1:  # Publish only once
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = TakeoffPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
