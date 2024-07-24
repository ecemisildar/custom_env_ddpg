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
        
    def send_request(self, model_name):
        # Read the SDF content from your file
        with open("/home/ei_admin/.gazebo/models/goal_circle/model.sdf", "r") as f:
            sdf_content = f.read()

        positions = {}  

        req = SpawnEntity.Request()
        req.name = f"{model_name}"
        req.xml = sdf_content
        req.robot_namespace = f"{model_name}"
        req.reference_frame = "world"

        # Generate random positions for each ball
        pose = Pose()
        
        pose.position.x = 3.0 #-2.0
        pose.position.y = 8.0 #8.0 
        pose.position.z = 4.0 #1.50
        req.initial_pose = pose

        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        positions[f"{model_name}"] = pos

        # Call the /spawn_entity service for each ball
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.success:
            self.get_logger().info(f'Successfully spawned model: {req.name}')
        else:
            self.get_logger().error(f'Failed to spawn model: {req.name}, {result.status_message}')  

        return positions   

def main(args=None):
    rclpy.init(args=args)

    spawn_client = GoalClient()
    model_name = input("Enter the base model name to spawn: ")  # Prompt for model name
    x = 0.0
    y = 0.0    
    spawn_client.send_request(model_name,x,y)
    
    spawn_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
     
