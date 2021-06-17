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
from skimage import io
from skimage.color import rgb2hsv, rgb2gray

from scipy.spatial import distance
import SpeedController as sc
from datetime import date
import datetime
import math
FRONT_WHEEL_RADIUS = 0.38
REAR_WHEEL_RADIUS = 0.6
WHEELBASE = 3

import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import cv2
import threading, queue
from queue import Empty

import pickle
import asyncio
import threading

# from NeuralNetwork import SignPrediction




def rotate(point, angle, origin = (0,0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class PathGenerator():
    def __init__(self, points:list, timestep = 50, max_point_distance = 5.0):
        super().__init__()
        self._timestep = 50
        self._speed = 0.0
        self._points = points
        self._max_point_distance = max_point_distance
        self._generator = self._path_generator()

    def _calculate_distance(self, x:tuple, y:tuple):
        return np.linalg.norm(np.array(x, dtype='float32') - np.array(y, dtype='float32'))

    @staticmethod
    def get_target_point(source, target, p):
        sx, sy = source
        tx, ty = target
        return sx + (tx - sx) * p, sy + (ty - sy) * p

    def _path_generator(self):
        inited = True
        source = target = (0,0)
        while len(self._points) > 0:
            if inited == True:
                source = self._points.pop(0)
                target = self._points.pop(0)
                inited = False
            else:
                source = target
                target = self._points.pop(0)
            
            

            distance_to_go = self._calculate_distance(source, target)
            distance_traveled = 0.0
            while distance_traveled < distance_to_go:
                distance_step = self._speed * (self._timestep / 1000)
                distance_traveled += distance_step
                precentage = distance_traveled / distance_to_go
                yield PathGenerator.get_target_point(source, target, precentage)
        
        while True:
            yield None
    
    def get_next_point(self, speed = 0.0, dist = 7.5):
        self._speed = speed 
        if dist > self._max_point_distance:
            self._speed = 0.0
        return next(self._generator)



class AckermannVehicleDriver():
    def __init__(self, graph = None, sign_recognition=False):
        #self.supervisor = Supervisor()

        self.driver = Driver() #self.supervisor.getFromDef('my_vehicle')
       
        self.keyboard = Keyboard()

        self.graph: Graph = graph


        # Extension Slot
        self.GPS = GPS('gps1'),# GPS('gps2'), GPS('gps3')
        self.gyro = Gyro('gyro')
        self.camera = Camera('camera')
        self.compass = Compass('compass')

        # Params
        self.speed: float = 0.0
        self.time_step = int(50)
        self.max_speed: float = 50.0
        self.steering_angle: float = 0.0
        self.max_steering_angle: float = 0.7 
        self.wheelbase = 3.0

        # Physic params
        self.weight = 500

        # Enable devices
        for gps in self.GPS: gps.enable(self.time_step)
        self.gyro.enable(self.time_step)
        self.keyboard.enable(self.time_step)
        self.camera.enable(self.time_step)
        self.compass.enable(self.time_step)
        # add speed controller
        self.speedController = sc.SpeedController(self.driver, self.GPS, self.gyro)
        # Constansts for steering
        max_theta_space = 100
        max_angles_space = 150
        self.theta_pi = np.linspace(0, 2*np.pi, max_theta_space)
        self.theta_cos = np.cos(self.theta_pi)
        self.theta_sin = np.sin(self.theta_pi)
        self.angles_space = np.linspace(-self.max_steering_angle, self.max_steering_angle, max_angles_space)
    
        self.radiuses = self.wheelbase / (np.tan(self.angles_space) + 1e-9) # +1e-9 to avoid div 0
        self.steering_iter = 0

        # Carrot position
        self.carrot_filter_size = 5
        self.carrot_position = np.zeros((self.carrot_filter_size, 2)) + np.inf
        self.max_point_distance = 5.0

        # Mean GPS filtering and physics data
        self.gps_xyz = []
        self.gps_speed = 0.0
        
        # CNN image data
        self.im_num = 0
        self.cnn_iter = 0
        self.cnn_per_iter = 5
        self.cnn_sem = threading.Semaphore(0)
        
        self._sp = None
        self.sign_recognition = sign_recognition
        if sign_recognition == True:
            self._sp = SignPrediction('classifierModGen.h5', 'unet2dMod.h5')
            self._run_sign_recognition_thread()


    def _compute_gps_xyz(self):
        xyz = np.zeros(3) #x, y, z = 0, 0, 0
        for gps in self.GPS:
            xyz += gps.getValues()
        self.gps_xyz = xyz / len(self.GPS)
    
    def _compute_gps_speed(self):
        speed = 0 
        for gps in self.GPS:
            speed += gps.getSpeed()
        self.gps_speed = speed / len(self.GPS)

    def _get_sign(self, num: float) -> int:
        return -1 if num <= 0.0 else 1

    def _set_steering_angle(self, angle: float):
        self.speedController.set_steering_angle(angle)



    def _set_speed(self, acceleration: float):
        self.speed += acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed:
            self.speed = -self.max_speed

        # self.driver.setCruisingSpeed(self.speed)
        self.speedController.set_expected_speed(acceleration)

    def follow_path(self, target_xy):
        object_xy = (self.gps_xyz[0], self.gps_xyz[2])
        if np.any(np.isnan(object_xy)):
            return False

        ox, oy = np.array(object_xy)
        tx, ty = np.array(target_xy)

        tx -= ox
        ty -= oy
        ox -= ox
        oy -= oy

        target_angle = math.atan2(-ty, -tx)
        cx, _, cy = self.compass.getValues()
        compass_angle = math.atan2(cy, cx)

        tx, ty = rotate((tx, ty), compass_angle - math.pi)

        dist = Graph.calculate_distance(object_xy, target_xy)
        radiuses_valid = np.abs(self.radiuses) > dist
        radiuses = self.radiuses[radiuses_valid]
        angles_space = self.angles_space[radiuses_valid]

        self.carrot_position[:-1] = self.carrot_position[1:]
        self.carrot_position[-1] = (tx, ty)
        carrot = self.carrot_position[np.where(np.sum(self.carrot_position, axis=1)) != np.inf][0]
        mean_tx, _ = np.mean(carrot, axis=0)

        phi = 0
        dist_min = abs(mean_tx)
        for r, phi_const in zip(radiuses, angles_space):
            rx = r*self.theta_cos + r
            ry = r*self.theta_sin
            xy = np.array(list(zip(rx,ry)))
            distance_rt = np.mean(distance.cdist(carrot, xy), axis=0).min()
            if dist_min > distance_rt:
                dist_min = distance_rt
                phi = phi_const

        if abs(phi) < 0.125: phi = 0.0
        self.driver.setSteeringAngle(phi)

    def _run_sign_recognition_thread(self):
        def _routine():
            while True:
                self.cnn_sem.acquire()
                im = rgb2gray(np.array(self.camera.getImageArray(), dtype='uint8'))
                to_unpack = self._sp.getSignIfExist(im)
                sign_id, sign_name, prob = to_unpack[0:3]
                
                if sign_id == -1: #brak znaku
                    pass
                if sign_id == 0: #stop
                    # TODO: do implementacji stop
                    pass
                if sign_id == 1: #przejscie dla pieszych
                    # TODO: do implementacji spowalnianie na przejściu dla pieszych
                    pass
                if sign_id == 2: #ograniczenie 20
                    # TODO: do implementacji ogranicznie do 20
                    pass
                if sign_id == 3: #ograniczenie 30
                    # TODO: do implementacji ogranicznie do 30
                    pass
                if sign_id == 4: #rondo
                    # TODO: do implementacji spowalnianie przy rondzie
                    pass

                print(sign_id, sign_name, prob)
        threading.Thread(target=_routine).start()


    def main_loop(self, xyz_target = None):
        while self.driver.step() != -1:
            self._compute_gps_xyz()
            self._compute_gps_speed()
            if not np.any(np.isnan(self.gps_xyz)) and not np.any(np.isnan(self.gps_speed)):
                break

        for i in range(5): 
            self._compute_gps_xyz()
            self.driver.step()

        print('Source: ', self.gps_xyz)
        print('Target: ', xyz_target)

        source = 'source'
        target = 'target'
        if xyz_target == None:
            xyz_target = [0, 0, 0]
        
        points, graph = self.graph.get_closest_point_edges_road(self.gps_xyz, xyz_target, source, target)

        print(points)

        point_list = []
        for i in range(len(points)):
            a = graph[points[i]]['xyz']        
            point_list.append((a[0], a[2]))
       
        if len(point_list) > 1:
            def func_dist(a, b, dist=5.0):
                d = Graph.calculate_distance(a, b)
                x = a[0] + dist/d * (b[0] - a[0])
                y = a[1] + dist/d * (b[1] - a[1])
                return (x, y)
            if Graph.calculate_distance(point_list[0], point_list[1]) > self.max_point_distance:
                point_list[0] = func_dist(point_list[0], point_list[1], self.max_point_distance)
            else:
                point_list = point_list[1:]

        road_point_gen = PathGenerator(point_list, self.time_step, self.max_point_distance)

        self._set_speed(30)
        target = (self.gps_xyz[0], self.gps_xyz[2])

        while self.driver.step() != -1:
            # self._update_display()
            self._compute_gps_xyz()
            self._compute_gps_speed()
            self.speedController.update_speed_controller(self.gps_speed)

            dist = Graph.calculate_distance((self.gps_xyz[0], self.gps_xyz[2]), target)
            target = road_point_gen.get_next_point(self.gps_speed, dist)
            if target is None:
                self.speedController.set_expected_speed(0.0)
                break


            self.follow_path(target)

            self.cnn_iter += 1
            if self.sign_recognition and self.cnn_iter % self.cnn_per_iter == 0:
                self.cnn_sem.release()
                
                
if __name__ == '__main__':
    #np.set_printoptions(suppress=True)

    args = sys.argv[1:]
    xyz_target = [float(x) for x in list(args[0].split(' '))]

    print('Time: ', datetime.datetime.now())
    world = Graph('../../world_map_new.json')

    driver = AckermannVehicleDriver(world, sign_recognition=False)

    driver.main_loop(xyz_target)

    

    
  
    
