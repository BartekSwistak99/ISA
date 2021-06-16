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
        self.time_step = int(50)
        self.driver = Driver()
        self.gps = GPS('gps')
        self.gyro=Gyro('gyro')
        self.speedController = sc.SpeedController(self.driver,self.gps,self.gyro)
        self.gps.enable(self.time_step)
        self.gyro.enable(self.time_step)
    def main_loop(self):
        while self.driver.step() != -1:
            self.speedController.update_speed_controller(self.gps.getSpeed())



if __name__ == '__main__':
    print('Time: ', datetime.datetime.now())
    driver = AckermannVehicleDriver()
    driver.speedController.set_expected_speed(20.0)
    # driver.speedController.set_steering_angle(-20.0)

    driver.main_loop()
