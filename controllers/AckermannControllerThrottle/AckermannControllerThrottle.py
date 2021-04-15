"""AckermannVehicleController controller."""

import datetime
import math

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import GPS, Gyro, Display
from controller import Keyboard
from controller import Motor
from vehicle import Driver


# FRONT_WHEEL_RADIUS = 0.38
# REAR_WHEEL_RADIUS = 0.6


def _get_sign(num: float) -> int:
    return -1 if num <= 0.0 else 1


class AckermannVehicleDriver:
    MAX_RPM = 6000  # defined in car model's, any change here will not affect model's mxa rpm
    SHIFT_RPM = 3000  # if actual rpm is bigger then SHIFT_RPM, car will shift to a higher gear
    DOWNSHIFT_RPM = 1000  # if actual rpm is smaller then SHIFT_RPM, car will downshift to lower gear
    ANGLE_PER_STEP = 0.035
    MAX_ANGLE = 0.5
    THROTTLE_PER_STEP = 15
    gear = 0
    max_gear: int
    rpm = 0
    current_speed = 0
    throttle = 0.0
    brake_intensity = 0.0
    expected_speed = 0.0
    MAX_SPEED = 180.0
    MIN_SPEED = -30.0

    def __init__(self):
        self.driver = Driver()
        self.keyboard = Keyboard()
        self.max_gear = self.driver.getGearNumber() - 1  # result of getGearNumber() include reverse ear, thus -1 is necessary

        print('max gears: ', str(self.max_gear))
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
        # Motors
        self.left_motor = Motor('right_front_wheel')
        self.right_motor = Motor('left_front_wheel')
        # Params
        self.time_step = int(50)
        self.steering_angle: float = 0.0
        self.max_steering_angle: float = 0.5

        # Enable devices
        self.GPS.enable(self.time_step)
        self.gyro.enable(self.time_step)
        self.keyboard.enable(self.time_step)
        self.driver.setThrottle(0)
        self.driver.setSteeringAngle(0.0)

    def _update_physics_params(self):
        self.rpm = self.driver.getRpm()
        self.current_speed = self.driver.getCurrentSpeed()
        self._set_gear()

    def _set_steering_angle(self, angle: float):
        sign = _get_sign(self.steering_angle)

        self.steering_angle += angle
        if self.steering_angle >= self.MAX_ANGLE:
            self.steering_angle = self.MAX_ANGLE
        if self.steering_angle <= -self.MAX_ANGLE:
            self.steering_angle = -self.MAX_ANGLE

        if sign == _get_sign(self.steering_angle):
            self.driver.setSteeringAngle(self.steering_angle)
        else:
            self.driver.setSteeringAngle(0.0)

    def _update_display(self):
        if self.display is None or self.im is None:
            return

        self.display.imagePaste(self.im, 0, 0)

        if math.isnan(self.current_speed): self.current_speed = 0.0
        if self.current_speed < 0.0: self.current_speed = abs(self.current_speed)

        alpha: float = self.current_speed / 260.0 * 3.72 - 0.27
        x = int(-self.needle_len * math.cos(alpha))
        y = int(-self.needle_len * math.sin(alpha))
        self.display.drawLine(100, 95, 100 + x, 95 + y)

        # # GPS speed
        # s = self.GPS.getSpeed() * 3.6
        # self.display.drawText('GPS speed: %.2f' % s, 10, 130)

        # gear
        self.display.drawText('Gear: %d' % self.gear, 10, 130)
        # rpm
        self.display.drawText('RPM: %d' % self.rpm, 75, 130)

        # Show driver speed
        self.display.drawText('Driver speed: %.2f' % self.current_speed, 10, 140)
        # Show driver speed
        # self.rpm =
        self.display.drawText('Driver speed: %.2f' % self.current_speed, 10, 140)

    def _set_throttle(self, acceleration: float, reverse=False):
        if self.rpm <= 100 and acceleration < 0.0:  # reverse gear on  - > shift to neutral if rpm is small enough
            self.gear = 0
            self.driver.setGear(self.gear)
            return
        if reverse and self.gear > 0.0:
            return
        elif not reverse and self.gear < 0.0 < acceleration:
            return

        self.throttle += acceleration / 100
        if self.throttle > 1.0:
            self.throttle = 1.0
        if self.throttle < 0.0:
            self.throttle = 0.0

        if self.throttle > 0.0 and self.gear == 0:
            if not reverse:
                self.gear = 1
                self.driver.setGear(self.gear)
            elif reverse:
                self.gear = -1
                self.driver.setGear(self.gear)

        self.driver.setThrottle(self.throttle)

    def _set_gear(self):
        if self.rpm > self.SHIFT_RPM and self.max_gear > self.gear > 0:  # shift gear only  between 0 (neutral) and 4 (max-1)
            self.gear += 1
            self.driver.setGear(self.gear)
            # print(self.gear)

        elif self.rpm < self.DOWNSHIFT_RPM and self.gear > 1:
            self.gear -= 1
            self.driver.setGear(self.gear)
            # print(self.gear)

    def _set_brake_intensity(self, intensity):
        self.brake_intensity = intensity
        if self.brake_intensity > 1.0:
            self.brake_intensity = 1.0
        elif self.brake_intensity < 0.0:
            self.brake_intensity = 0.0

        self.driver.setBrakeIntensity(self.brake_intensity)

    def set_expected_speed(self, expected_speed):
        self.expected_speed = expected_speed
        if self.expected_speed < self.MIN_SPEED:
            self.expected_speed = self.MIN_SPEED
        elif self.expected_speed > self.MAX_SPEED:
            self.expected_speed = self.MAX_SPEED

        self.expected_speed = expected_speed
    # def _update_speed(self):
        # if self.expected_speed >

    def _check_keyboard(self):
        key = self.keyboard.getKey()
        if key == 32 and self.brake_intensity < 1.0:
            print('stop')
            self._set_brake_intensity(1.0)
        elif self.brake_intensity > 0:
            self._set_brake_intensity(0.0)

        if key == Keyboard.LEFT:
            self._set_steering_angle(-self.ANGLE_PER_STEP)
        if key == Keyboard.RIGHT:
            self._set_steering_angle(+self.ANGLE_PER_STEP)

        if key == Keyboard.UP:
            self._set_throttle(self.THROTTLE_PER_STEP)
        elif key == Keyboard.DOWN:
            self._set_throttle(self.THROTTLE_PER_STEP / 3, reverse=True)
        else:
            self._set_throttle(-self.THROTTLE_PER_STEP)

    def main_loop(self):
        while self.driver.step() != -1:
            self._update_physics_params()
            self._check_keyboard()
            self._update_display()
            pass

if __name__ == '__main__':
    print('Time: ', datetime.datetime.now())

    driver = AckermannVehicleDriver()

    driver.main_loop()
