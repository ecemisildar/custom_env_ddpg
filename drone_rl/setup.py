from setuptools import find_packages, setup

package_name = 'drone_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ei_admin',
    maintainer_email='ecemisildar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawn_drone = drone_rl.spawn_drone:main',
            'spawn_entities = drone_rl.spawn_entities:main',
            'delete_entities = drone_rl.delete_entites:main',
            'goal_position = drone_rl.goal_position:main',
            'drone_env = drone_rl.drone_env:main',
            'drone_env2 = drone_rl.drone_env2:main',
            'takeoff_publisher = drone_rl.takeoff:main',
        ],
    },
)
