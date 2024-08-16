import os
import yaml
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    use_gui = DeclareLaunchArgument("use_gui", default_value="false", choices=["true", "false"],
                                    description="Whether to execute gzclient")
    xacro_file_name = "sjtu_drone2.urdf.xacro"
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    xacro_file = os.path.join(
        get_package_share_directory("sjtu_drone_description"),
        "urdf", xacro_file_name
    )
    yaml_file_path = os.path.join(
        get_package_share_directory('drone_rl'),
        'config', 'drone2.yaml'
    )   
    
    robot_description_config = xacro.process_file(xacro_file, mappings={"params_path": yaml_file_path})
    robot_desc = robot_description_config.toxml()
    # get ns from yaml
    model_ns = "drone_2"


    # def launch_gzclient(context, *args, **kwargs):
    #     if context.launch_configurations.get('use_gui') == 'false':
    #         return [IncludeLaunchDescription(
    #             PythonLaunchDescriptionSource(
    #                 os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
    #             ),
    #             launch_arguments={'verbose': 'true'}.items()
    #         )]
    #     return []

    return LaunchDescription([
        use_gui,
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace=model_ns,
            output="screen",
            parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_desc, "frame_prefix": model_ns + "/"}],
            arguments=[robot_desc]
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace=model_ns,
            output='screen',
        ),

       

        Node(
            package="drone_rl",
            executable="spawn_drone",
            namespace=model_ns,
            arguments=[robot_desc, model_ns, "2", "0", "10"],
            output="screen"
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            namespace=model_ns,
            arguments=["2", "0", "10", "0", "0", "0", "world", f"{model_ns}/odom"],
            output="screen"
        ),
    ])
