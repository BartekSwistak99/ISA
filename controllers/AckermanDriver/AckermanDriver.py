"""AckermannVehicleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, GPS, Gyro, Display
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

        self.flag = True
        self.flag1 = False

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

    def _check_keyboard(self):
        key = self.keyboard.getKey()
        if key == Keyboard.LEFT:
            self._set_steering_angle(-0.1)
        if key == Keyboard.RIGHT:
            self._set_steering_angle(+0.1)

        if key == Keyboard.UP:
            self._set_speed(+0.5)
        if key == Keyboard.DOWN:
            self._set_speed(-0.5)

        if key == 32:
            print('stop')
            self.driver.setBrakeIntensity(1.0)

        if key == -1:
            sign = 1 if self.speed < 0 else -1
            speed_ms = self.speed / 3.6

            kinetic_energy = 0.5 * self.weight * (speed_ms ** 2)
            force_air = 0.5 * 0.5 * 1.21 * 1.4 * speed_ms**2
            rolling_resistance = 0.02 * self.weight * 9.81 * 1

            force_total = force_air + rolling_resistance
            d = kinetic_energy / force_total

            a = 0 if d == 0 else speed_ms**2 / (2 * d)

            if abs(speed_ms) < 0.1:
                self._set_speed(-self.speed)
            else:
                self._set_speed(sign*a)

            # m = 500
            # g = 9.81
            # theta = 0
            # v = x || 50
            # Crr = wheelsDampingConstant || 0.02

            # Rolling resistance = Crr * Normal_force
            # Normal force = m * g * cos(theta)

            # RR = 0.02 * 500 * 9.81 * 1
            # RR = 98,1N

            # Fop = 0.5 * C * p * S * v^2
            # C = 0.5
            # S = 1.4m^2
            # p = 1.21kg/m^3
            # Fop = 0.5 * 0.5 * 1.21 * 1.4 * 13.9^2
            # Fop = 82,32N

            # F = 180,4N

            # theta = 0 stopni

            # Work = Force * displacement * cosine(Theta)
            # 0.5*m*v^2 = F*d
            # ke = 0.5 * m * v^2

            # v = 50 km/h = 13,9 m/s
            # m = 500 kg
            # ke = 0,5 * 500 kg * (13,9 m/s) ^ 2 = 250 kg * 192,90 m/s = 48225,30 kg*m^2/s^2
            # F = 1 kg*m/s^2
            # d = ke / F =  48225,30kg*m^2/s^2 / 1kg*m/s^2 = 48225,30m
            #

    def main_loop(self):
        while self.driver.step() != -1:
            self._check_keyboard()
            self._update_display()
            pass


if __name__ == '__main__':
    print('Time: ', datetime.datetime.now())

    #controller = AckermannVehicleController()
    # controller.main_loop()

    driver = AckermannVehicleDriver()
    driver.main_loop()
