"""epuck_new_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

print('Start epuck_new_controller')
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(10.0)
rightMotor.setPosition(10.0)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
i = 0
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    # val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    leftMotor.setPosition(i)
    rightMotor.setPosition(i)
    i += 1
    pass

# Enter here exit cleanup code.
