import cv2, dlib
import numpy as np
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

img = np.load('img.npy') # or use cv2.imread

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class Points2D:
    def __init__(self, img):
        self.face = predictor(img, detector(img)[0])
    def _get(self, i):
        return [self.face.part(i).x, self.face.part(i).y]
    def points(self):
        vertices = [30, 8, 36, 45, 48, 54]
        return np.array(list(map(self._get, vertices)), dtype='float')

def process_face(face):

    ''' skeleton from learnopencv.com / yinguobing head pose estimation '''
    face_model_points = np.array([
        [   0.0,    0.0,    0.0],
        [   0.0, -330.0, - 65.0],
        [-225.0,  170.0, -135.0],
        [ 225.0,  170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [ 150.0, -150.0, -125.0]])

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    focal_length = face.shape[1]
    center = (face.shape[1]/2, face.shape[0]/2)

    camera_matrix = np.array([
        [focal_length, 1, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype='float')

    dist_coefs = np.zeros((4, 1))
    image_points = Points2D(face).points()

    success, rotation_vector, translation_vector =\
        cv2.solvePnP(face_model_points, image_points,
                     camera_matrix, dist_coefs)

    return rotation_vector

rotation = process_face(img) # pitch roll yaw
points = Points2D(img).points()

plt.imshow(img)
plt.scatter(points[:, 0], points[:, 1])