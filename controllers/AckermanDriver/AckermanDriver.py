"""AckermannVehicleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, GPS, Gyro, Display, Camera
from controller import Keyboard
from vehicle import Driver

import numpy as np
from scipy.ndimage.interpolation import shift

from datetime import date
import datetime
import math
FRONT_WHEEL_RADIUS = 0.38
REAR_WHEEL_RADIUS = 0.6


class AckermannVehicleDriver():
    def __init__(self):
        self.driver = Driver()
        self.keyboard = Keyboard()

        # Display
        self.im = None
        self.display = Display('display')
        if self.display is not None:
            self.im = self.display.imageLoad('speedometer.png')
        self.needle_len = 50.0

        # GPS
        self.GPS = GPS('gps')

        # Gyro
        self.gyro = Gyro('gyro')

        # Camera
        self.camera = Camera('camera')

        # Params
        self.speed: float = 0.0
        self.time_step = int(50)
        self.max_speed: float = 50.0
        self.steering_angle: float = 0.0
        self.max_steering_angle: float = 0.5

        # Physic params
        self.weight = 500

        # Enable devices
        self.GPS.enable(self.time_step)
        self.gyro.enable(self.time_step)
        self.keyboard.enable(self.time_step)
        self.camera.enable(self.time_step)

        # Auto drive
        self.SIZE_LINE = 3
        self.old_line_values = np.zeros(self.SIZE_LINE)
        self.old_value = 0.0
        self.integral = 0.0


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

    def _update_display(self):
        if self.display is None or self.im is None:
            return

        self.display.imagePaste(self.im, 0, 0)

        current_speed: float = self.driver.getCurrentSpeed()
        if math.isnan(current_speed):
            current_speed = 0.0
        if current_speed < 0.0:
            current_speed = abs(current_speed)

        alpha: float = current_speed / 260.0 * 3.72 - 0.27
        x = int(-self.needle_len * math.cos(alpha))
        y = int(-self.needle_len * math.sin(alpha))
        self.display.drawLine(100, 95, 100 + x, 95 + y)

        # GPS speed
        s = self.GPS.getSpeed() * 3.6
        self.display.drawText('GPS speed: %.2f' % s, 10, 130)

        # Show driver speed
        self.display.drawText('Driver speed: %.2f' % current_speed, 10, 140)

    def _set_speed(self, acceleration: float):
        self.speed += acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed:
            self.speed = -self.max_speed

        self.driver.setCruisingSpeed(self.speed)
        # 100km/h 0 -> 100 => 1 sekund
        # setCruisingSpeed(0) 100..0  => 1 sekund
        #
    
    def _check_camera(self):
        image = self.camera.getImageArray()
        data = np.array(image, dtype='uint8')
        data = np.absolute(data - [255, 255, 0])
        
        sums = np.sum(data, axis=2)
        threshold = np.where(data < 30)
        pixs = threshold[0].shape[0]
        sumx = np.sum(threshold[0])


        magic = 0
        if pixs != 0:
            magic = (sumx / pixs / self.camera.getWidth() - 0.5) * self.camera.getFov()

        shift(self.old_line_values, 1)

        self.old_line_values[self.SIZE_LINE - 1] = magic
        yellow_line_angle = np.sum(self.old_line_values) / self.SIZE_LINE

        diff = yellow_line_angle - self.old_value
        self.old_value = yellow_line_angle

        angle = 0.05 * yellow_line_angle + diff
        print(angle)
        self._set_steering_angle(angle)

    def main_loop(self):
        self._set_speed(5)
        while self.driver.step() != -1:
            self._check_camera()
            self._update_display()
            pass


if __name__ == '__main__':
    print('Time: ', datetime.datetime.now())

    #controller = AckermannVehicleController()
    # controller.main_loop()

    driver = AckermannVehicleDriver()
    driver.main_loop()
