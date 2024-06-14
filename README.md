# DDPG to Control Drone

The agent learns how to collect goal points in a custom environment in Gazebo using the DDPG algorithm and image observation

## Environment Description
The custom environment created in the Gazebo simulation is a square area with 4 walls and green ball objects spawning at random positions in every episode. The drone starts at the origin with a random orientation on the z-axis. 
 ## Observation Space 
 The drone agent only uses image observations thanks to its RGB camera
```
self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height,self.width, self.channel), dtype=np.uint8)
```
## Action Space
The drone agent takes velocity linear actions on the x-axis and angular velocity actions on the z-axis
```
self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float16)
```

## Done 
The episode is done when the agent collects all the target points in the environment

## Reward Function
The reward function for 1 target point is given below:
```
reward = -1 + 2/(1+np.exp(self.dist_array[0]/(self.num_green_pixels+1)))
```
It is a logistic function which combines distances from the target and the green pixel number that the agent sees

![Reward function plot](https://github.com/ecemisildar/custom_env_ddpg/blob/main/figure_1.png)

## Termination
Termination is possible when the drone hits the walls or exceeds the episode time

## Publishers
To publish velocity, takeoff and landing commands respectively
```
self.speed_motors_pub = self.node.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
self.takeoff_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/takeoff', 10)
self.land_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/land', 10)
```
## Subscribers
Subscription to the camera, odometry and collision topics respectively
```
self.image_sub = self.node.create_subscription(Image, '/simple_drone/front/image_raw', self.camera_callback, 10)
self.current_pose_sub = self.node.create_subscription(Odometry, '/simple_drone/odom', self.position_callback, 10)
self.collision_sub = self.node.create_subscription(ContactsState, '/simple_drone/bumper_states', self.collision_callback, 10)
```        
## Services
To call my custom clients reset world and spawn target points respectively
```
self.reset_client = self.node.create_client(Empty, '/reset_world')
self.spawn_client = self.node.create_client(SpawnEntity, '/spawn_entity')
```

# Demo Video

![Demo Video](https://github.com/ecemisildar/custom_env_ddpg/blob/main/gif_example.gif)



The drone model is taken from: https://github.com/NovoG93/sjtu_drone

DDPG algorithm sources:
https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/tensorflow2/pendulum
https://github.com/samkoesnadi/DDPG-tf2/tree/master

