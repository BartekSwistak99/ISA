"""AckermannVehicleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard
from vehicle import Driver

from datetime import date
import datetime
import math
FRONT_WHEEL_RADIUS = 0.38
REAR_WHEEL_RADIUS = 0.6

class AckermannVehicleDriver():
    def __init__(self):
        self.driver = Driver()
        self.keyboard = Keyboard()
        
        self.time_step = int(50)
        self.keyboard.enable(self.time_step)
        # Params
        self.speed: float = 2.0
        self.max_speed: float = 50.0
        self.steering_angle: float = 0.0
        self.max_steering_angle: float = 0.5
    
    def _get_sign(self, num: float) -> int:
        return -1 if num <= 0.0 else 1

    def _set_steering_angle(self, angle: float):
        sign = self._get_sign(self.steering_angle)

        self.steering_angle += angle
        if self.steering_angle >= self.max_steering_angle:
            self.steering_angle = self.max_steering_angle
        if self.steering_angle <= -self.max_steering_angle:
            self.steering_angle = -self.max_steering_angle
        
        if sign == self._get_sign(self.steering_angle):
            self.driver.setSteeringAngle(self.steering_angle)
        else: 
            self.driver.setSteeringAngle(0.0)

        
    def _set_speed(self, acceleration: float):
        self.speed += acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed:
            self.speed = -self.max_speed
            
        self.driver.setCruisingSpeed(self.speed)
        
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
        while self.driver.step() != -1:
            self._check_keyboard()
            pass
        
if __name__ == '__main__':
    print('Time: ', datetime.datetime.now()) 

    #controller = AckermannVehicleController()
    #controller.main_loop()
    
    driver = AckermannVehicleDriver()
    driver.main_loop()
    
    

