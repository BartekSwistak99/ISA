from skimage import io
from skimage.morphology import skeletonize, dilation, square
from glob import glob
import numpy as np
import matplotlib.pyplot as pyplot
import math
from scipy.spatial import distance

import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte, io
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import cv2 as cv

def rotate(point, angle, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def detect_sign():
    img = cv.imread('znak5.png',0)
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,50,
                            param1=75,param2=30,minRadius=20,maxRadius=60)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv.imshow('detected circles',cimg)
    cv.waitKey(0)

if __name__ == '__main__':

    theta_pi = np.linspace(0, 2*np.pi, 250)
    theta_cos = np.cos(theta_pi)
    theta_sin = np.sin(theta_pi)

    L = 3
    max_angles_space = 150
    angle_max = 0.7
    object_xy = np.array([0, 0])
    #target_xy = np.array([0, 0])
    target_xy = np.array([1 ,1])

    target_xy -= object_xy
    object_xy -= object_xy
    ox, oy = object_xy
    tx, ty = target_xy



    #tx, ty = rotate([tx, ty], np.pi/4)

    print('dot', np.dot(object_xy, target_xy))

    distance_ot = np.sqrt((ox-tx)**2 + (oy-ty)**2)

    # Obliczamy max_space możliwych trajektori
    angles_space = np.linspace(-angle_max, angle_max, max_angles_space)

    # Obliczamy promień ze wzoru
    radiuses = L / (np.tan(angles_space) + 1e-5)

    # Usuwamy promienie, których rozmiar jest mniejszy niż dystans między punktami
    radiuses_valid = np.abs(radiuses) > distance_ot
    radiuses = radiuses[radiuses_valid]
    angles_space = angles_space[radiuses_valid]

    print(radiuses.size)
    phi = 0.0
    rx_min, ry_min = None, None
    dist_min = abs(tx)
    print(dist_min)

    for r, a in zip(radiuses, angles_space):
        rx = r*theta_cos + r
        ry = r*theta_sin
        xy = np.array(list(zip(rx, ry)))
        distance_rt = distance.cdist([[tx, ty]], xy).min()
        if dist_min > distance_rt:
            dist_min = distance_rt
            phi = a

            rx_min, ry_min = rx, ry
        #pyplot.plot(rx, ry)
    pyplot.plot(rx_min, ry_min)

    #pyplot.plot(rx_min, ry_min)
    print(phi)

    #Scrircle = np.array([(x + r, y) for r in radiuses])

    pyplot.plot([ox], [oy], marker='o', markersize=5, color='red')
    pyplot.plot([tx], [ty], marker='o', markersize=5, color='blue')
    pyplot.plot([ox, tx], [oy, ty])
    pyplot.show()

    pass
