"""AckermannVehicleController controller."""

import datetime
import math

import SpeedController as sc
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import GPS, Gyro, Display
from controller import Keyboard
from controller import Motor
from vehicle import Driver


# FRONT_WHEEL_RADIUS = 0.38
# REAR_WHEEL_RADIUS = 0.6




# asdasd

class AckermannVehicleDriver:

    def __init__(self):
        self.driver = Driver()
        self.speedController = sc.SpeedController(self.driver)

    def main_loop(self):
        while self.driver.step() != -1:
            self.speedController.update_speed_controller()
            if self.speedController.current_speed > 20 and self.speedController.gear == -1:
                self.speedController.set_expected_speed(80.0)
            pass


if __name__ == '__main__':
    print('Time: ', datetime.datetime.now())
    driver = AckermannVehicleDriver()
    driver.speedController.set_expected_speed(-20.0)

    driver.main_loop()
