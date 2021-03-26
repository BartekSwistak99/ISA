"""AckermannVehicleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard

import math
FRONT_WHEEL_RADIUS = 0.38
REAR_WHEEL_RADIUS = 0.6

class AckermannVehicleController():
    def __init__(self):
        # Main class
        self.robot = Robot()
        self.keyboard = Keyboard()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.keyboard.enable(self.time_step)

        # Wheels
        self.left_front_wheel = self.robot.getDevice("left_front_wheel")
        self.right_front_wheel = self.robot.getDevice("right_front_wheel")
        self.left_rear_wheel = self.robot.getDevice("left_rear_wheel")
        self.right_rear_wheel = self.robot.getDevice("right_rear_wheel")

        self.wheels = [self.left_front_wheel, self.right_front_wheel,
        self.left_rear_wheel, self.right_rear_wheel]
        for wheel in self.wheels:
            wheel.setPosition(math.inf)

        # Steers
        self.left_steer = self.robot.getDevice("left_steer")
        self.right_steer = self.robot.getDevice("right_steer")
        self.steers = [self.left_steer, self.right_steer]

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()

        # Params
        self.speed: float = 5.0
        self.max_speed: float = 40.0
        self.steering_angle: float = 0.0
        self.max_steering_angle: float = 1.8

        
    def _set_speed(self, acceleration: float):
        self.speed += acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed:
            self.speed = -self.max_speed
        
        front_ang_vel = self.speed * 1000.0 / 3600.0 / FRONT_WHEEL_RADIUS
        rear_ang_vel = self.speed * 1000.0 / 3600.0 / REAR_WHEEL_RADIUS

        for front, rear in zip(self.wheels[0:2], self.wheels[2:]):
            front.setVelocity(front_ang_vel)
            rear.setVelocity(rear_ang_vel)

    def _set_steering_angle(self, angle: float):
        self.steering_angle += angle
        if self.steering_angle >= self.max_steering_angle:
            self.steering_angle = self.max_steering_angle
        if self.steering_angle <= -self.max_steering_angle:
            self.steering_angle = -self.max_steering_angle
        
        for steer in self.steers:
            steer.setPosition(angle)    

    def _check_keyboard(self):
        key = self.keyboard.getKey()
        if key == Keyboard.LEFT:
            self._set_steering_angle(-0.1)
        if key == Keyboard.RIGHT:
            self._set_steering_angle(+0.1)

        if key == Keyboard.UP:
            self._set_speed(+0.25)
        if key == Keyboard.DOWN:
            self._set_speed(-0.25)

        if key == -1:
            sign = 1 if self.speed < 0 else -1
            self._set_speed(sign * math.sqrt(abs(self.speed)))
    
    def main_loop(self):
        while self.robot.step(self.time_step) != -1:
            self._check_keyboard()
            pass

if __name__ == '__main__':
    print('Controller started')

    controller = AckermannVehicleController()
    controller.main_loop()
    
    

