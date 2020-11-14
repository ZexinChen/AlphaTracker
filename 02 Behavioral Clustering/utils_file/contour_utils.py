import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import os

def load_json(json_name):

    # json_name= '/Users/ruihan/Documents/clustering/results/0603/1929_black_two/alphapose-results-forvis-tracked.json'

    with open(json_name,'r') as json_file:
        data = json.loads(json_file.read())

    return data



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)



def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    _,phi1 = cart2pol(v1[0],v1[1])
    phi1 = math.degrees(phi1)

    _,phi2 = cart2pol(v2[0],v2[1])
    phi2 = math.degrees(phi2)


    if (phi1<phi2):
        if (phi2-phi1)<180:
            flag = True
        else:
            flag = False
    else:
        if (phi1-phi2)<180:
            flag = False
        else:
            flag = True
    if flag:
        angle = - angle

    return angle, np.cos(angle),np.sin(angle)



def pose_to_feature(pose):

    feature = []
    bad_frame = False

    pose = np.asarray(pose)
    if pose.size<12:
        bad_frame = True
        return bad_frame,feature

    pose = np.reshape(pose,(4,3))
    # if len(pose)<4:
    #     bad_frame = True
    #     return bad_frame,feature

    nose,ear1,ear2,tail = pose[0,0:2], pose[1,0:2], pose[2,0:2], pose[3,0:2]

    head = nose - (ear1+ear2)/2
    body = nose - tail
    if (np.linalg.norm(head) == 0) or (np.linalg.norm(body) == 0) :
        bad_frame = True
        return bad_frame,feature

    feature += [
        np.linalg.norm(nose-tail),
        np.linalg.norm(nose-ear1),
        np.linalg.norm(nose-ear2)
    ]

    angle, cos_angle,sin_angle = angle_between(body,head)
    feature += [
        cos_angle,
        sin_angle,
    ]

    return bad_frame,feature


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if os.path.exists(path) and os.path.isdir(path):
            pass
        else: raise

