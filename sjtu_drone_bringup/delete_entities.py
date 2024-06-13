import rclpy
from geometry_msgs.msg import Pose
import numpy as np
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity

  
class DeleteEntityClient(Node):

    def __init__(self):
        super().__init__('delete_entities')
        self.cli = self.create_client(DeleteEntity, '/delete_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
    def send_request(self, entity_name):
        self.req = DeleteEntity.Request()
        self.req.name = entity_name
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.success:
            self.get_logger().info(f'Successfully deleted entity: {entity_name}')
        else:    
            self.get_logger().error(f'Failed to delete entity: {entity_name}, {result.status_message}')

        return future.result()

def main(args=None):
    rclpy.init(args=args)

    delete_client = DeleteEntityClient()
    entity_name = input("Enter the entity name to delete: ")  # Prompt for entity name
    
   
    delete_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  