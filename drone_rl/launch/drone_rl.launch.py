import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    drone_rl_path = get_package_share_directory('drone_rl')
    
    yaml_file_path = os.path.join(
        get_package_share_directory('drone_rl'),
        'config', 'drone.yaml'
    )

    model_ns = "drone"

    with open(yaml_file_path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        model_ns = yaml_dict["namespace"]

    return LaunchDescription([

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(drone_rl_path, 'launch', 'drone_gazebo.launch.py')
            )
        ),    
    ])

if __name__ == '__main__':
    generate_launch_description()