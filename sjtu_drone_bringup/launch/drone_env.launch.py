import launch
from launch_ros.actions import Node
from launch.substitutions import LaunchSubstitution
from launch import LaunchDescription

def generate_launch_description():
    world_file = LaunchSubstitution(value="$(find sjtu_drone_description)/world/env_and_balls.world")
    use_sim_time = LaunchSubstitution(value="true")  # Adjust as needed
    gui = LaunchSubstitution(value="true")  # Adjust as needed
    headless = LaunchSubstitution(value="false")  # Adjust as needed
    debug = LaunchSubstitution(value="false")  # Adjust as needed

    return LaunchDescription([
        # Start Gazebo server with the specified world
        Node(
            package='gazebo_ros',
            executable='gzserver',
            name='gazebo',
            arguments=[world_file],
            remappings=[
                # Remap '/clock' to '/sim_time' if Gazebo doesn't publish sim time by default
                ('clock', '/sim_time')  # Optional, adjust as needed
            ],
            output='screen'
        ),

        # Spawn the drone model using spawn_entity.py
        Node(
            package='sjtu_drone_bringup',
            executable='spawn_entity.py',  # Assuming you have a script here
            name='drone_spawner',
            output='screen',
            arguments=[
                '-entity', 'drone',  # Replace with your model name if different
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.5',
                '-urdf',  # Use -urdf if your model is a URDF file
                '-param', 'robot_description'  # Reference the robot description parameter
            ]
        ),

        # Launch your air_drone node
        Node(
            package='sjtu_drone_bringup',
            executable='drone_env.py',
            name='drone_env',
            output='screen'
        ),
    ])
