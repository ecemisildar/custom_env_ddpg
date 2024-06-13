import rclpy
from geometry_msgs.msg import Pose
import numpy as np
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity



class SpawnEntityClient(Node):

    def __init__(self):
        super().__init__('spawn_entities')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
    def send_request(self, model_name, count):
        # Read the SDF content from your file
        with open("/home/ei_admin/.gazebo/models/green_ball_0/model.sdf", "r") as f:
            sdf_content = f.read()

        positions = {}    

        for idx in range(count):
            req = SpawnEntity.Request()
            req.name = f"{model_name}_{idx}"
            req.xml = sdf_content
            req.robot_namespace = f"{model_name}_{idx}"
            req.reference_frame = "world"

            # Generate random positions for each ball
            random_position = np.random.uniform(-5, 5, size=3)
            pose = Pose()
            pose.position.x = np.random.uniform(-5, 5)
            pose.position.y = np.random.uniform(-5, 5)
            # pose.position.x = 2.0
            # pose.position.y = 0.0
            pose.position.z = 0.8
            req.initial_pose = pose

            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            positions[f"{model_name}_{idx}"] = pos
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

    spawn_client = SpawnEntityClient()
    model_name = input("Enter the base model name to spawn: ")  # Prompt for model name
    count = 1  # Number of balls to spawn

    spawn_client.send_request(model_name, count)
    
    spawn_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
     
