"""AckermannVehicleController controller."""

import datetime
import threading

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import GPS, Gyro
from vehicle import Driver

import SpeedController as sc


# FRONT_WHEEL_RADIUS = 0.38
# REAR_WHEEL_RADIUS = 0.6


# asdasd

class AckermannVehicleDriver:

	def __init__(self):
		self.time_step = int(50)
		self.driver = Driver()
		self.gps = GPS('gps')
		self.gyro = Gyro('gyro')
		self.speedController = sc.SpeedController(self.driver, self.gps, self.gyro)
		self.gps.enable(self.time_step)
		self.gyro.enable(self.time_step)

	def main_loop(self):
		while self.driver.step() != -1:
			self.speedController.update_speed_controller(self.gps.getSpeed())
		# print(f'{self.speedController.brake_intensity} ')


if __name__ == '__main__':
	print('Time: ', datetime.datetime.now())
	driver = AckermannVehicleDriver()
	driver.speedController.set_expected_speed(30.0)
	threading.Timer(5, driver.speedController.slow_down, [0.0]).start()

	driver.main_loop()
