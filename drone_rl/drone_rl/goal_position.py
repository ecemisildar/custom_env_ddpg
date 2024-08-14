import rclpy
from geometry_msgs.msg import Pose
import numpy as np
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
import time

class GoalClient(Node):

    def __init__(self):
        super().__init__('goal_position')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
    def send_request(self, model_name, x, y, z):
        # Read the SDF content from your file
        sdf_path = "/home/ei_admin/.gazebo/models/goal_circle_blue/model.sdf"
        try:
            with open(sdf_path, "r") as f:
                sdf_content = f.read()
        except FileNotFoundError:
            self.get_logger().error(f"SDF file not found at {sdf_path}")
            return {}
        
        req = SpawnEntity.Request()
        req.name = model_name
        req.xml = sdf_content
        req.robot_namespace = model_name
        req.reference_frame = "world"

        # Set initial pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        req.initial_pose = pose

        # Call the /spawn_entity service
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        
        if result.success:
            self.get_logger().info(f'Successfully spawned model: {model_name}')
        else:
            self.get_logger().error(f'Failed to spawn model: {model_name}, {result.status_message}')  

        return {model_name: np.array([x, y, z])}

def main(args=None):
    rclpy.init(args=args)
    goal_client = GoalClient()
    
    model_name = 'goal_circle_blue'
    x, y, z = -1.0, 8.0, 11.0
    
    # Use the send_request method with specific data
    data = goal_client.send_request(model_name, x, y, z)
    print(data)
    
    goal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
