"""AckermannVehicleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, GPS, Gyro, Display, Camera, Supervisor, Compass
from controller import Keyboard
from vehicle import Driver

import sys
sys.path.append('../')
from Dijkstra import Graph
sys.path.pop()

from scipy.ndimage.interpolation import shift
from skimage.morphology import skeletonize, dilation, square
from glob import glob
import numpy as np


from datetime import date
import datetime
import math
FRONT_WHEEL_RADIUS = 0.38
REAR_WHEEL_RADIUS = 0.6


class AckermannVehicleDriver():
    def __init__(self, graph = None):
        #self.supervisor = Supervisor()

        self.driver = Driver() #self.supervisor.getFromDef('my_vehicle')
       
        self.keyboard = Keyboard()

        self.graph = graph

        # Display
        self.im = None
        self.display = Display('display')
        if self.display is not None:
            self.im = self.display.imageLoad('speedometer.png')
        self.needle_len = 50.0

        # Extension Slot
        self.GPS = GPS('gps')
        self.gyro = Gyro('gyro')
        self.camera = Camera('camera')
        self.compass = Compass('compass')

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
        self.compass.enable(self.time_step)

        # Auto drive
        self.SIZE_LINE = 2
        self.old_line_values = np.zeros(self.SIZE_LINE)
        self.old_value = 0.0
        self.integral = 0.0


    def _get_sign(self, num: float) -> int:
        return -1 if num <= 0.0 else 1

    def _set_steering_angle(self, angle: float):
        sign = self._get_sign(self.steering_angle)

        self.steering_angle = angle
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
    
    def _check_camera(self):
        image = self.camera.getImageArray()
        data = np.array(image, dtype='uint8')
        data = np.absolute(data - [255, 255, 0])
        
        sums = np.sum(data, axis=2)
        morf = sums < 210
        morf = dilation(morf, selem=square(5))
        morf = skeletonize(morf)

        threshold = np.where(morf == 1)
        pixs = threshold[0].shape[0]
        if pixs < 30:
            return

        sumx = np.sum(threshold[0])
        magic = (sumx / pixs / self.camera.getWidth() - 0.5) * self.camera.getFov()

        shift(self.old_line_values, 1)

        self.old_line_values[self.SIZE_LINE - 1] = magic
        yellow_line_angle = np.sum(self.old_line_values) / self.SIZE_LINE

        diff = yellow_line_angle - self.old_value
        self.old_value = yellow_line_angle


        speed = self.GPS.getSpeed() * 3.6
        ax = 5.0
        if speed > 7.5 and speed < 200.0:
            ax = speed * 1.15
        # if speed > 20.0 and speed < 200.0:
        #     ax = speed * 1.75

        angle = (ax / 1000) * yellow_line_angle + diff
        #print(f"{angle:+2.5f}, {ax:+2.5f}, {speed:+2.5f}", end='\r')
        self._set_steering_angle(angle)
    
    def follow_path(self, source, target, points, min_dist = 1.5):
        if len(points) == 0: 
            print('Reached target location')
            return True

        object_xyz = self.GPS.getValues()
        target_xyz = self.graph.get_points_coord(points[0])
        dist = Graph.calculate_distance(object_xyz, target_xyz)
        if  dist < min_dist:
            print('Reached: ', points[0])
            points.pop(0)
            return False

        dy = object_xyz[2] - target_xyz[2] 
        dx = object_xyz[0] - target_xyz[0]
        target_angle = math.atan2(dy, dx)

        x, _, y = self.compass.getValues()
        object_angle = math.atan2(x, y)

        radians = object_angle - target_angle

        if radians > +math.pi: radians = math.pi - radians
        if radians < -math.pi: radians = 2*math.pi + radians

        if abs(radians) > (math.pi - 0.025): radians = 0.0

        print(target_angle, object_angle, radians)
        if not math.isnan(radians):
            self._set_steering_angle(radians)

        return False


    def main_loop(self):
        source = 'J'
        target = 'AN'
        points:tuple = self.graph.dijkstra_shortest(source, target)

        print(points, self.graph.get_points_coord('J')) 
        
        self._set_speed(10)
        while self.driver.step() != -1:
            #self._check_camera()
            self._update_display()

            if self.follow_path(source, target, points):
                self.driver.setCruisingSpeed(0)
                break

if __name__ == '__main__':
    print('Time: ', datetime.datetime.now())

    world = Graph('../../world_map.json')

    driver = AckermannVehicleDriver(world)
    driver.main_loop()

    

    
  
    
