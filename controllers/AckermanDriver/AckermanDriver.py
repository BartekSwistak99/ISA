"""AckermannVehicleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, GPS, Gyro, Display, Camera, Supervisor, Compass
from controller import Keyboard
from vehicle import Driver

import sys
sys.path.append('../')
from Dijkstra import Graph
from CNN import load_model, translate_network_to_sign_to_word
sys.path.pop()

from scipy.ndimage.interpolation import shift
from skimage.morphology import skeletonize, dilation, square
from glob import glob
import numpy as np
from skimage import io

from scipy.spatial import distance

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


class CNNQueue():
    def __init__(self, network_path):
        self.model = load_model(network_path)
        self.img_q = queue.Queue()
        self.task_done = queue.Queue()
    
    def run_worker(self):
        def worker():
            while True:
                image_data = self.img_q.get()
                #prediction = self.model.predict(image_data)

                print(image_data)
                
                try:
                    while True:
                        self.img_q.get_nowait()
                except Empty:
                    continue
                self.img_q.task_done()

        threading.Thread(target=worker, daemon=True).start()
    
    def put_image_data(self, image_data):
        self.img_q.put(image_data)


    def get_image_prediction(self):
        if self.task_done.qsize == 0:
            return [np.zeros(6)]
        return self.task_done.get()
    



class AckermannVehicleDriver():
    def __init__(self, graph = None):
        #self.supervisor = Supervisor()

        self.driver = Driver() #self.supervisor.getFromDef('my_vehicle')
       
        self.keyboard = Keyboard()

        self.graph: Graph = graph

        # Display
        self.im = None
        self.display = Display('display')
        if self.display is not None:
            self.im = self.display.imageLoad('speedometer.png')
        self.needle_len = 50.0

        # Extension Slot
        self.GPS = GPS('gps1'), #GPS('gps2'), GPS('gps3')
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

        # Auto drive
        self.SIZE_LINE = 2
        self.old_line_values = np.zeros(self.SIZE_LINE)
        self.old_value = 0.0
        self.integral = 0.0

        # Constansts for steering
        max_theta_space = 50
        max_angles_space = 150
        self.theta_pi = np.linspace(0, 2*np.pi, max_theta_space)
        self.theta_cos = np.cos(self.theta_pi)
        self.theta_sin = np.sin(self.theta_pi)
        self.angles_space = np.linspace(-self.max_steering_angle, self.max_steering_angle, max_angles_space)
    
        self.radiuses = self.wheelbase / (np.tan(self.angles_space) + 1e-9) # +1e-9 to avoid div 0
        self.steering_iter = 0

        # Mean GPS filtering
        self.gps_xyz = []
        self.gps_speed = 0.0

        # CNN image data
        self.im_num = 0


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
        s = self.gps_speed * 3.6
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
    
    def follow_path(self, source, target, points, graph, min_dist = 2.0):
        if len(points) == 0: 
            print('Reached target location')
            return True

        
        object_xyz = self.gps_xyz
        if np.any(np.isnan(object_xyz)):
            return False
        
        target_xyz = graph[points[0]]['xyz']
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

        radians = (object_angle - target_angle)

        if radians > +math.pi: radians = math.pi - radians
        if radians < -math.pi: radians = 2*math.pi + radians

        if abs(radians) > (math.pi - 0.125): radians = 0.0

        func_val = math.log10(dist + 1) ** 4
        if func_val > 1.0: func_val = 1.0
        radians *= func_val

        if not math.isnan(radians):
            self._set_steering_angle(radians)

        return False
    

    @staticmethod
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

    def follow_path_ackerman(self, source, target, points, graph, min_dist = 2.5):
        if len(points) == 0: 
            print('Reached target location')
            return True

        
        object_xyz = self.gps_xyz
        if np.any(np.isnan(object_xyz)):
            return False

        target_xyz = graph[points[0]]['xyz']
        dist = Graph.calculate_distance(object_xyz, target_xyz)
        if  dist < min_dist:
            print('Reached: ', points[0])
            points.pop(0)
            return False
        
        ox, _, oy = np.array(object_xyz)
        tx, _, ty = np.array(target_xyz)

        tx -= ox
        ty -= oy
        ox -= ox
        oy -= oy

        #ty = np.abs(ty)

        target_angle = math.atan2(-ty, -tx)
        cx, _, cy = self.compass.getValues()
        compass_angle = math.atan2(cy, cx)

        radians = (compass_angle + target_angle)

        if radians > +math.pi: radians = math.pi - radians
        if radians < -math.pi: radians = 2*math.pi + radians


        tx, ty = AckermannVehicleDriver.rotate((tx, ty), compass_angle - math.pi)

        radiuses_valid = np.abs(self.radiuses) > dist
        radiuses = self.radiuses[radiuses_valid]
        angles_space = self.angles_space[radiuses_valid]

        phi = 0
        dist_min = abs(tx)
        for r, phi_const in zip(radiuses, angles_space):
            rx = r*self.theta_cos + r
            ry = r*self.theta_sin
            xy = np.array(list(zip(rx,ry)))
            distance_rt = distance.cdist([[tx,ty]], xy).min()
            if dist_min > distance_rt:
                dist_min = distance_rt
                phi = phi_const

        
        if self.steering_iter % 5 == 0:
            self.driver.setSteeringAngle(phi)
        self.steering_iter += 1

        #self._set_steering_angle(phi)
        return False


    def follow_path_ackerman_better(self, points, min_dist = 2.5):
        if len(points) == 0: 
            print('Reached target location')
            return True

        
        object_xyz = self.gps_xyz
        if np.any(np.isnan(object_xyz)):
            return False

        target_xyz = points[0]
        dist = Graph.calculate_distance(object_xyz, target_xyz)
        if  dist < min_dist:
            print('Reached: ', points[0])
            points.pop(0)
            return False
        
        ox, _, oy = np.array(object_xyz)
        tx, _, ty = np.array(target_xyz)

        tx -= ox
        ty -= oy
        ox -= ox
        oy -= oy

        #ty = np.abs(ty)

        target_angle = math.atan2(-ty, -tx)
        cx, _, cy = self.compass.getValues()
        compass_angle = math.atan2(cy, cx)

        radians = (compass_angle + target_angle)

        if radians > +math.pi: radians = math.pi - radians
        if radians < -math.pi: radians = 2*math.pi + radians


        tx, ty = AckermannVehicleDriver.rotate((tx, ty), compass_angle - math.pi)

        radiuses_valid = np.abs(self.radiuses) > dist
        radiuses = self.radiuses[radiuses_valid]
        angles_space = self.angles_space[radiuses_valid]

        phi = 0
        dist_min = abs(tx)
        for r, phi_const in zip(radiuses, angles_space):
            rx = r*self.theta_cos + r
            ry = r*self.theta_sin
            xy = np.array(list(zip(rx,ry)))
            distance_rt = distance.cdist([[tx,ty]], xy).min()
            if dist_min > distance_rt:
                dist_min = distance_rt
                phi = phi_const

        
        func_val = math.log10(dist + 1) ** 2
        if func_val > 1.0: func_val = 1.0
        #phi *= func_val

        #if phi < 0.1: phi = 0

        #if self.steering_iter % 5 == 0:
        self.driver.setSteeringAngle(phi)
        self.steering_iter += 1

        #self._set_steering_angle(phi)
        return False


    def save_camera_image(self):
        image = self.camera.getImageArray()
        data = np.array(image, dtype='uint8')
        
        im_save_path = f'F:/Deep Learning/Data/webots/{self.im_num}_5_0.png'
        #io.imsave(im_save_path, data)

        self.im_num += 1
        #print(self.im_num)
        if self.im_num == 750:
            exit()

    def predict_camera(self, model):
        image = self.camera.getImageArray()
        data = np.array(image, dtype='uint8') / 255.0
        data = np.array([data])
        print(translate_network_to_sign_to_word(model.predict(data)[0]))
    

    def main_loop(self, xyz_target = None):
        # Load CNN model
        #model = load_model('C:/Users/adrian/Documents/Semestr_6/ISA/ISA/model_sign.h5')


        # wait for gps 
        while self.driver.step() != -1:
            self._compute_gps_xyz()
            self._compute_gps_speed()
            if not np.any(np.isnan(self.gps_xyz)):
                break

        print('Source: ', self.gps_xyz)
        print('Target: ', xyz_target)

        source = 'source'
        target = 'target'
        if xyz_target == None:
            xyz_target = [0, 0, 0]
        
        points, graph = self.graph.get_closest_point_edges_road(self.gps_xyz, xyz_target, source, target)

        new_list_xyz = []
        for i in range(len(points) - 1):
            a = graph[points[i]]['xyz']
            b = graph[points[i + 1]]['xyz']
            space = np.linspace(a, b, 5)
            for s in space:
                new_list_xyz.append(s)
        


        self._set_speed(7)
        while self.driver.step() != -1:
            self._update_display()
            self._compute_gps_xyz()
            self._compute_gps_speed()

            
            #print(cnn.get_image_prediction())
            #self.predict_camera(model)
            #self.save_camera_image()
            #self.detect_circle()


            if self.follow_path_ackerman_better(new_list_xyz):
            #if self.follow_path(source, target, points, graph):
               self.driver.setCruisingSpeed(0)
               break




if __name__ == '__main__':
    #np.set_printoptions(suppress=True)

    args = sys.argv[1:]
    xyz_target = [float(x) for x in list(args[0].split(' '))]

    print('Time: ', datetime.datetime.now())
    world = Graph('../../world_map_new.json')

    driver = AckermannVehicleDriver(world)
    driver.main_loop(xyz_target)

    

    
  
    
