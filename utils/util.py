import os
from collections import namedtuple, defaultdict
import numpy as np
from threading import Thread, Lock, Condition
import cv2
import time
import g2o
from g2o.contrib import SmoothEstimatePropagator
from numbers import Number
from queue import Queue
from enum import Enum
from utils.GRAPH import *
from utils.optimization import *

class ImageReader(object):
    def __init__(self, ids, timestamps, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10  # 10 images ahead of current index
        self.waiting = 1.5  # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)

    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue

            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape

class KITTI_dataset_reader(object):
    '''path example: 'path/to/your/KITTI odometry dataset/sequences/00'''
    def __init__(self, path):
        Cam = namedtuple('cam', 'fx fy cx cy width height baseline')
        cam00_02 = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376, 0.5371657)
        cam03 = Cam(721.5377, 721.5377, 609.5593, 172.854, 1241, 376, 0.53715)
        cam04_12 = Cam(707.0912, 707.0912, 601.8873, 183.1104, 1241, 376, 0.53715)

        path = os.path.expanduser(path)
        timestamps = np.loadtxt(os.path.join(path, 'times.txt'))
        self.left = ImageReader(self.listdir(os.path.join(path, 'image_0')),timestamps)
        self.right = ImageReader(self.listdir(os.path.join(path, 'image_1')),timestamps)

        assert len(self.left) == len(self.right)
        self.timestamps = self.left.timestamps

        sequence = int(path.strip(os.path.sep).split(os.path.sep)[-1])
        if sequence < 3:
            self.cam = cam00_02
        elif sequence == 3:
            self.cam = cam03
        elif sequence < 13:
            self.cam = cam04_12

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.left)

class Camera(object):
    def __init__(self, fx, fy, cx, cy, w, h,
                 frustum_near, frustum_far, baseline):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.baseline = baseline
        self.K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        self.frustum_near = frustum_near
        self.frustum_far = frustum_far

        self.width = w
        self.height = h

    def compute_right_camera_pose(self, pose):
        pos = pose * np.array([self.baseline, 0, 0])
        return g2o.Isometry3d(pose.orientation(), pos)

#----------------------------------------------------------------------
def row_match(matcher, kps1, desps1, kps2, desps2,
        matching_distance=40, max_row_distance=2.5, max_disparity=100):
    #matches = matcher.match(np.array(desps1, dtype=np.float32), np.array(desps2, dtype=np.float32))
    matches = matcher.match(np.array(desps1), np.array(desps2))
    good = []
    for m in matches:
        pt1 = kps1[m.queryIdx].pt
        pt2 = kps2[m.trainIdx].pt
        if (m.distance < matching_distance and
            abs(pt1[1] - pt2[1]) < max_row_distance and
            abs(pt1[0] - pt2[0]) < max_disparity):   # epipolar constraint
            good.append(m)
    return good

def circular_stereo_match(matcher, desps1, desps2, matches12,
        desps3, desps4, matches34, matching_distance=30,min_matches=10, ratio=0.8):
    dict_m13, dict_m24 = dict(),dict()
    dict_m34 = dict((m.queryIdx, m) for m in matches34)
    ms13 = matcher.knnMatch(np.array(desps1), np.array(desps3), k=2)
    for (m, n) in ms13:
        if m.distance < min(matching_distance, n.distance * ratio):
            dict_m13[m.queryIdx] = m
    # to avoid unnecessary computation
    if len(dict_m13) < min_matches:
        return []
    ms24 = matcher.knnMatch(np.array(desps2), np.array(desps4), k=2)
    for (m, n) in ms24:
        if m.distance < min(matching_distance, n.distance * ratio):
            dict_m24[m.queryIdx] = m

    matches = []
    for m in matches12:
        shared13 = dict_m13.get(m.queryIdx, None)
        shared24 = dict_m24.get(m.trainIdx, None)

        if shared13 is not None and shared24 is not None:
            shared34 = dict_m34.get(shared13.trainIdx, None)
            if (shared34 is not None and shared34.trainIdx == shared24.trainIdx):
                matches.append((shared13, shared24))
    return matches

class ImageFeature(object):
    def __init__(self, image, params):
        self.image = image
        self.height, self.width = image.shape[:2]

        self.keypoints = []      # list of cv2.KeyPoint
        self.descriptors = []    # numpy.ndarray

        self.detector = params.feature_detector
        self.extractor = params.descriptor_extractor
        self.matcher = params.descriptor_matcher

        self.cell_size = params.matching_cell_size
        self.distance = params.matching_distance
        self.neighborhood = (params.matching_cell_size * params.matching_neighborhood)
        self._lock = Lock()

    def extract(self):
        self.keypoints = self.detector.detect(self.image)
        self.keypoints, self.descriptors = self.extractor.compute(self.image, self.keypoints)
        self.unmatched = np.ones(len(self.keypoints), dtype=bool)

    def draw_keypoints(self, name='keypoints', delay=1):
        if self.image.ndim == 2:
            image = np.repeat(self.image[..., np.newaxis], 3, axis=2)
        else:
            image = self.image
        img = cv2.drawKeypoints(image, self.keypoints, None, flags=0)
        cv2.imshow(name, img);cv2.waitKey(delay)

    def find_matches(self, predictions, descriptors):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))
        for m, query_idx, train_idx in self.matched_by(descriptors):
            if m.distance > min(distances[train_idx], self.distance):
                continue

            pt1 = predictions[query_idx]
            pt2 = self.keypoints[train_idx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx*dx + dy*dy) > self.neighborhood:
                continue

            matches[train_idx] = query_idx
            distances[train_idx] = m.distance
        matches = [(i, j) for j, i in matches.items()]
        return matches

    def matched_by(self, descriptors):
        with self._lock:
            unmatched_descriptors = self.descriptors[self.unmatched]
            if len(unmatched_descriptors) == 0:
                return []
            lookup = dict(zip(range(len(unmatched_descriptors)),
                np.where(self.unmatched)[0]))

        #matches = self.matcher.match(np.array(descriptors,dtype=np.float32), np.asarray(unmatched_descriptors, dtype=np.float32))
        matches = self.matcher.match(np.array(descriptors), unmatched_descriptors)

        return [(m, m.queryIdx, m.trainIdx) for m in matches]

    def row_match(self, *args, **kwargs):
        return row_match(self.matcher, *args, **kwargs)

    def circular_stereo_match(self, *args, **kwargs):
        return circular_stereo_match(self.matcher, *args, **kwargs)

    def get_keypoint(self, i):
        return self.keypoints[i]

    def get_descriptor(self, i):
        return self.descriptors[i]

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width-1))
        y = int(np.clip(pt[1], 0, self.height-1))
        color = self.image[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def set_matched(self, i):
        with self._lock:
            self.unmatched[i] = False

    def get_unmatched_keypoints(self):
        keypoints = []
        descriptors = []
        indices = []

        with self._lock:
            for i in np.where(self.unmatched)[0]:
                keypoints.append(self.keypoints[i])
                descriptors.append(self.descriptors[i])
                indices.append(i)

        return keypoints, descriptors, indices

class Frame(object):
    def __init__(self, idx, pose, feature, cam, timestamp=None, pose_covariance=np.identity(6)):
        self.idx = idx
        self.pose = pose  # g2o.Isometry3d
        self.feature = feature
        self.cam = cam
        self.timestamp = timestamp
        self.image = feature.image

        self.orientation = pose.orientation()
        self.position = pose.position()
        self.pose_covariance = pose_covariance

        self.transform_matrix = pose.inverse().matrix()[:3]  # shape: (3, 4)
        self.P = (self.cam.K.dot(self.transform_matrix))  # from world frame to image

    # batch version
    def can_view(self, points, ground=False, margin=20):  # Frustum Culling
        points = np.transpose(points)
        (u, v), depth = self.project(self.transform(points))

        if ground:
            return np.logical_and.reduce([
                depth >= self.cam.frustum_near,
                depth <= self.cam.frustum_far,
                u >= - margin,
                u <= self.cam.width + margin])
        else:
            return np.logical_and.reduce([
                depth >= self.cam.frustum_near,
                depth <= self.cam.frustum_far,
                u >= - margin,
                u <= self.cam.width + margin,
                v >= - margin,
                v <= self.cam.height + margin])

    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose
        self.orientation = self.pose.orientation()
        self.position = self.pose.position()
        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.P = (self.cam.K.dot(self.transform_matrix))

    def transform(self, points):  # from world coordinates
        '''Transform points from world coordinates frame to camera frame.
        Args: points: a point or an array of points, of shape (3,) or (3, N).'''
        R = self.transform_matrix[:3, :3]
        if points.ndim == 1:
            t = self.transform_matrix[:3, 3]
        else:
            t = self.transform_matrix[:3, 3:]
        return R.dot(points) + t

    def project(self, points):
        '''Project points from camera frame to image's pixel coordinates.
        Args:points: a point or an array of points, of shape (3,) or (3, N).
        Returns:Projected pixel coordinates, and respective depth.'''
        projection = self.cam.K.dot(points / points[-1:])
        return projection[:2], points[-1]

    def find_matches(self, points, descriptors):
        '''Match to points from world frame.
        Args:points: a list/array of points. shape: (N, 3)
            descriptors: a list of feature descriptors. length: N
        Returns:List of successfully matched (queryIdx, trainIdx) pairs.'''
        points = np.transpose(points)
        proj, _ = self.project(self.transform(points))
        proj = proj.transpose()
        return self.feature.find_matches(proj, descriptors)

    def get_keypoint(self, i):
        return self.feature.get_keypoint(i)

    def get_descriptor(self, i):
        return self.feature.get_descriptor(i)

    def get_color(self, pt):
        return self.feature.get_color(pt)

    def set_matched(self, i):
        self.feature.set_matched(i)

    def get_unmatched_keypoints(self):
        return self.feature.get_unmatched_keypoints()

class StereoFrame(Frame):
    def __init__(self, idx, pose, feature, right_feature, cam,
                 right_cam=None, timestamp=None, pose_covariance=np.identity(6)):

        super().__init__(idx, pose, feature, cam, timestamp, pose_covariance)
        self.left = Frame(idx, pose, feature, cam, timestamp, pose_covariance)
        self.right = Frame(idx, cam.compute_right_camera_pose(pose),
                           right_feature, right_cam or cam, timestamp, pose_covariance)

    def find_matches(self, source, points, descriptors):
        q2 = Queue()
        def find_right(points, descriptors, q):
            m = dict(self.right.find_matches(points, descriptors))
            q.put(m)

        t2 = Thread(target=find_right, args=(points, descriptors, q2))
        t2.start()
        matches_left = dict(self.left.find_matches(points, descriptors))
        t2.join()
        matches_right = q2.get()

        measurements = []
        for i, j in matches_left.items():
            if i in matches_right:
                j2 = matches_right[i]

                y1 = self.left.get_keypoint(j).pt[1]
                y2 = self.right.get_keypoint(j2).pt[1]
                if abs(y1 - y2) > 2.5:  # epipolar constraint
                    continue  # TODO: choose one

                meas = Measurement(Measurement.Type.STEREO,
                    source,[self.left.get_keypoint(j),
                     self.right.get_keypoint(j2)],
                    [self.left.get_descriptor(j),self.right.get_descriptor(j2)])
                measurements.append((i, meas))
                self.left.set_matched(j)
                self.right.set_matched(j2)
            else:
                meas = Measurement(Measurement.Type.LEFT,
                    source,[self.left.get_keypoint(j)],[self.left.get_descriptor(j)])
                measurements.append((i, meas))
                self.left.set_matched(j)

        for i, j in matches_right.items():
            if i not in matches_left:
                meas = Measurement(Measurement.Type.RIGHT,
                    source,[self.right.get_keypoint(j)],
                    [self.right.get_descriptor(j)])
                measurements.append((i, meas))
                self.right.set_matched(j)

        return measurements

    def match_mappoints(self, mappoints, source):
        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)
        matched_measurements = self.find_matches(source, points, descriptors)

        measurements = []
        for i, meas in matched_measurements:
            meas.mappoint = mappoints[i]
            measurements.append(meas)
        return measurements

    def triangulate(self):
        kps_left, desps_left, idx_left = self.left.get_unmatched_keypoints()
        kps_right, desps_right, idx_right = self.right.get_unmatched_keypoints()

        mappoints, matches = self.triangulate_points(
            kps_left, desps_left, kps_right, desps_right)

        measurements = []
        for mappoint, (i, j) in zip(mappoints, matches):
            meas = Measurement(Measurement.Type.STEREO,Measurement.Source.TRIANGULATION,
                [kps_left[i], kps_right[j]],[desps_left[i], desps_right[j]])
            meas.mappoint = mappoint
            meas.view = self.transform(mappoint.position)
            measurements.append(meas)
            self.left.set_matched(idx_left[i])
            self.right.set_matched(idx_right[j])

        return mappoints, measurements

    def triangulate_points(self, kps_left, desps_left, kps_right, desps_right):
        matches = self.feature.row_match(
            kps_left, desps_left, kps_right, desps_right)
        assert len(matches) > 0

        px_left = np.array([kps_left[m.queryIdx].pt for m in matches])
        px_right = np.array([kps_right[m.trainIdx].pt for m in matches])

        points = cv2.triangulatePoints(
            self.left.P,self.right.P,
            px_left.transpose(),px_right.transpose()
        ).transpose()  # shape: (N, 4)
        points = points[:, :3] / points[:, 3:]
        can_view = np.logical_and(
            self.left.can_view(points),
            self.right.can_view(points))
        mappoints,matchs = [],[]
        for i, point in enumerate(points):
            if not can_view[i]:
                continue
            normal = point - self.position
            normal = normal / np.linalg.norm(normal)
            color = self.left.get_color(px_left[i])

            mappoint = MapPoint(point, normal, desps_left[matches[i].queryIdx], color)
            mappoints.append(mappoint)
            matchs.append((matches[i].queryIdx, matches[i].trainIdx))

        return mappoints, matchs

    def update_pose(self, pose):
        super().update_pose(pose)
        self.right.update_pose(pose)
        self.left.update_pose(
            self.cam.compute_right_camera_pose(pose))

    # batch version
    def can_view(self, mappoints):
        points = []
        point_normals = []
        for i, p in enumerate(mappoints):
            points.append(p.position)
            point_normals.append(p.normal)
        points = np.asarray(points)
        point_normals = np.asarray(point_normals)

        normals = points - self.position
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        cos = np.clip(np.sum(point_normals * normals, axis=1), -1, 1)
        parallel = np.arccos(cos) < (np.pi / 4)

        can_view = np.logical_or(
            self.left.can_view(points),
            self.right.can_view(points))

        return np.logical_and(parallel, can_view)

    def to_keyframe(self):
        return KeyFrame(self.idx, self.pose,
            self.left.feature, self.right.feature,
            self.cam, self.right.cam,
            self.pose_covariance)

class MapPoint(GraphMapPoint):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, normal, descriptor,
                 color=np.zeros(3), covariance=np.identity(3) * 1e-4):
        super().__init__()
        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1
        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color
        # self.owner = None
        self.count = defaultdict(int)

    def update_position(self, position):
        self.position = position

    def update_normal(self, normal):
        self.normal = normal

    def update_descriptor(self, descriptor):
        self.descriptor = descriptor

    def set_color(self, color):
        self.color = color

    def is_bad(self):
        with self._lock:
            status = (self.count['meas'] == 0
                    or (self.count['outlier'] > 20
                        and self.count['outlier'] > self.count['inlier'])
                    or (self.count['proj'] > 20
                        and self.count['proj'] > self.count['meas'] * 10))
            return status

    def increase_outlier_count(self):
        with self._lock:
            self.count['outlier'] += 1

    def increase_inlier_count(self):
        with self._lock:
            self.count['inlier'] += 1

    def increase_projection_count(self):
        with self._lock:
            self.count['proj'] += 1

    def increase_measurement_count(self):
        with self._lock:
            self.count['meas'] += 1

class Measurement(GraphMeasurement):
    Source = Enum('Measurement.Source', ['TRIANGULATION', 'TRACKING', 'REFIND'])
    Type = Enum('Measurement.Type', ['STEREO', 'LEFT', 'RIGHT'])

    def __init__(self, type, source, keypoints, descriptors):
        super().__init__()

        self.type = type
        self.source = source
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.view = None  # mappoint's position in current coordinates frame

        self.xy = np.array(self.keypoints[0].pt)
        if self.is_stereo():
            self.xyx = np.array([
                *keypoints[0].pt, keypoints[1].pt[0]])

        self.triangulation = (source == self.Source.TRIANGULATION)

    def get_descriptor(self, i=0):
        return self.descriptors[i]

    def get_keypoint(self, i=0):
        return self.keypoints[i]

    def get_descriptors(self):
        return self.descriptors

    def get_keypoints(self):
        return self.keypoints

    def is_stereo(self):
        return self.type == Measurement.Type.STEREO

    def is_left(self):
        return self.type == Measurement.Type.LEFT

    def is_right(self):
        return self.type == Measurement.Type.RIGHT

    def from_triangulation(self):
        return self.triangulation

    def from_tracking(self):
        return self.source == Measurement.Source.TRACKING

    def from_refind(self):
        return self.source == Measurement.Source.REFIND

class KeyFrame(GraphKeyFrame, StereoFrame):
    _id = 0
    _id_lock = Lock()

    def __init__(self, *args, **kwargs):
        GraphKeyFrame.__init__(self)
        StereoFrame.__init__(self, *args, **kwargs)

        with KeyFrame._id_lock:
            self.id = KeyFrame._id
            KeyFrame._id += 1

        self.reference_keyframe = None
        self.reference_constraint = None
        self.preceding_keyframe = None
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def update_reference(self, reference=None):
        if reference is not None:
            self.reference_keyframe = reference
        self.reference_constraint = (
            self.reference_keyframe.pose.inverse() * self.pose)

    def update_preceding(self, preceding=None):
        if preceding is not None:
            self.preceding_keyframe = preceding
        self.preceding_constraint = (
            self.preceding_keyframe.pose.inverse() * self.pose)

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed

#------------------------------------------------------------------------
class MotionModel(object):
    def __init__(self,timestamp=None,initial_position=np.zeros(3),
                 initial_orientation=g2o.Quaternion(),initial_covariance=None):
        self.timestamp = timestamp
        self.position = initial_position
        self.orientation = initial_orientation
        self.covariance = initial_covariance  # pose covariance

        self.v_linear = np.zeros(3)  # linear velocity
        self.v_angular_angle = 0
        self.v_angular_axis = np.array([1, 0, 0])

        self.initialized = False
        # damping factor
        self.damp = 0.95

    def current_pose(self):
        '''Get the current camera pose.'''
        return (g2o.Isometry3d(self.orientation, self.position),
                self.covariance)

    def predict_pose(self, timestamp):
        '''Predict the next camera pose.'''
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position),
                    self.covariance)

        dt = timestamp - self.timestamp
        delta_angle = g2o.AngleAxis(
            self.v_angular_angle * dt * self.damp,
            self.v_angular_axis)
        delta_orientation = g2o.Quaternion(delta_angle)
        position = self.position + self.v_linear * dt * self.damp
        orientation = self.orientation * delta_orientation
        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp,new_position, new_orientation, new_covariance=None):
        '''Update the motion model when given a new camera pose.'''
        if self.initialized:
            dt = timestamp - self.timestamp
            assert dt != 0

            v_linear = (new_position - self.position) / dt
            self.v_linear = v_linear

            delta_q = self.orientation.inverse() * new_orientation
            delta_q.normalize()

            delta_angle = g2o.AngleAxis(delta_q)
            angle = delta_angle.angle()
            axis = delta_angle.axis()

            if angle > np.pi:
                axis = axis * -1
                angle = 2 * np.pi - angle

            self.v_angular_axis = axis
            self.v_angular_angle = angle / dt

        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation
        self.covariance = new_covariance
        self.initialized = True

    def apply_correction(self, correction):  # corr: g2o.Isometry3d or matrix44
        '''Reset the model given a new camera pose.
        Note: This method will be called when it happens an abrupt change in the pose (LoopClosing)'''
        if not isinstance(correction, g2o.Isometry3d):
            correction = g2o.Isometry3d(correction)
        current = g2o.Isometry3d(self.orientation, self.position)
        current = current * correction
        self.position = current.position()
        self.orientation = current.orientation()
        self.v_linear = (correction.inverse().orientation() * self.v_linear)
        self.v_angular_axis = (correction.inverse().orientation() * self.v_angular_axis)

#------------------------------------------------------------------------
class Mapping(object):
    def __init__(self, graph, params):
        self.graph = graph
        self.params = params
        self.local_keyframes = []

        self.optimizer = LocalBA()

    def add_keyframe(self, keyframe, measurements):
        self.graph.add_keyframe(keyframe)
        self.create_points(keyframe)

        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

        self.local_keyframes.clear()
        self.local_keyframes.append(keyframe)

        self.fill(self.local_keyframes, keyframe)
        self.refind(self.local_keyframes, self.get_owned_points(keyframe))

        self.bundle_adjust(self.local_keyframes)
        self.points_culling(self.local_keyframes)

    def fill(self, keyframes, keyframe):
        covisible = sorted(
            keyframe.covisibility_keyframes().items(),
            key=lambda _: _[1], reverse=True)

        for kf, n in covisible:
            if n > 0 and kf not in keyframes and self.is_safe(kf):
                keyframes.append(kf)
                if len(keyframes) >= self.params.local_window_size:
                    return

    def create_points(self, keyframe):
        mappoints, measurements = keyframe.triangulate()
        self.add_measurements(keyframe, mappoints, measurements)

    def add_measurements(self, keyframe, mappoints, measurements):
        for mappoint, measurement in zip(mappoints, measurements):
            self.graph.add_mappoint(mappoint)
            self.graph.add_measurement(keyframe, mappoint, measurement)
            mappoint.increase_measurement_count()

    def bundle_adjust(self, keyframes):
        adjust_keyframes = set()
        for kf in keyframes:
            if not kf.is_fixed():
                adjust_keyframes.add(kf)

        fixed_keyframes = set()
        for kf in adjust_keyframes:
            for ck, n in kf.covisibility_keyframes().items():
                if (n > 0 and ck not in adjust_keyframes
                        and self.is_safe(ck) and ck < kf):
                    fixed_keyframes.add(ck)

        self.optimizer.set_data(adjust_keyframes, fixed_keyframes)
        completed = self.optimizer.optimize(self.params.ba_max_iterations)

        self.optimizer.update_poses()
        self.optimizer.update_points()

        if completed:
            self.remove_measurements(self.optimizer.get_bad_measurements())
        return completed

    def is_safe(self, keyframe):
        return True

    def get_owned_points(self, keyframe):
        owned = []
        for m in keyframe.measurements():
            if m.from_triangulation():
                owned.append(m.mappoint)
        return owned

    def filter_unmatched_points(self, keyframe, mappoints):
        filtered = []
        for i in np.where(keyframe.can_view(mappoints))[0]:
            pt = mappoints[i]
            if (not pt.is_bad() and
                    not self.graph.has_measurement(keyframe, pt)):
                filtered.append(pt)
        return filtered

    def refind(self, keyframes, new_mappoints):  # time consuming
        if len(new_mappoints) == 0:
            return
        for keyframe in keyframes:
            filtered = self.filter_unmatched_points(keyframe, new_mappoints)
            if len(filtered) == 0:
                continue
            for mappoint in filtered:
                mappoint.increase_projection_count()

            measuremets = keyframe.match_mappoints(filtered, Measurement.Source.REFIND)

            for m in measuremets:
                self.graph.add_measurement(keyframe, m.mappoint, m)
                m.mappoint.increase_measurement_count()

    def remove_measurements(self, measurements):
        for m in measurements:
            m.mappoint.increase_outlier_count()
            self.graph.remove_measurement(m)

    def points_culling(self, keyframes):  # Remove bad mappoints
        mappoints = set(chain(*[kf.mappoints() for kf in keyframes]))
        for pt in mappoints:
            if pt.is_bad():
                self.graph.remove_mappoint(pt)

class MappingThread(Mapping):
    def __init__(self, graph, params):
        super().__init__(graph, params)

        self._requests_cv = Condition()
        self._requests = [False, False]  # requests: [LOCKWINDOW_REQUEST, PROCESS_REQUEST]

        self._lock = Lock()
        self.locked_window = set()
        self.status = defaultdict(bool)

        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def add_keyframe(self, keyframe, measurements):
        self.graph.add_keyframe(keyframe)

        self.create_points(keyframe)
        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

        self._queue.put(keyframe)
        with self._requests_cv:
            self._requests_cv.notify()

    def maintenance(self):
        stopped = False
        while not stopped:
            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    stopped = True
                    self._requests[1] = True
                    break
                else:
                    self.local_keyframes.append(keyframe)
                    if len(self.local_keyframes) >= 5:
                        self._requests[1] = True
                        break

            with self._requests_cv:
                if self._requests.count(True) == 0:
                    self._requests_cv.wait()

                    while not self._queue.empty():
                        keyframe = self._queue.get()
                        if keyframe is None:
                            stopped = True
                            self._requests[1] = True
                            break
                        else:
                            self.local_keyframes.append(keyframe)
                            if len(self.local_keyframes) >= 5:
                                self._requests[1] = True

                requests = self._requests[:]
                self._requests[0] = False
                self._requests[1] = False

            self.status['processing'] = True

            if requests[1] and len(self.local_keyframes) > 0:
                self.fill(self.local_keyframes, self.local_keyframes[-1])

            if requests[0]:
                with self._lock:
                    for kf in self.local_keyframes:
                        self.locked_window.add(kf)
                        for ck, n in kf.covisibility_keyframes().items():
                            if n > 0:
                                self.locked_window.add(ck)
                    self.status['window_locked'] = True

            if requests[1] and len(self.local_keyframes) > 0:
                completed = self.bundle_adjust(self.local_keyframes)
                if completed:
                    self.points_culling(self.local_keyframes)
                self.local_keyframes.clear()

            self.status['processing'] = False

    def stop(self):
        with self._requests_cv:
            self._requests_cv.notify()

        while not self._queue.empty():
            time.sleep(1e-4)
        self._queue.put(None)  # sentinel value
        self.maintenance_thread.join()
        print('mapping stopped')

    def is_safe(self, keyframe):
        with self._lock:
            return not self.is_window_locked() or keyframe in self.locked_window

    def is_processing(self):
        return self.status['processing']

    def lock_window(self):
        with self._lock:
            self.status['window_locked'] = False
            self.locked_window.clear()

        with self._requests_cv:
            self._requests[0] = True
            self._requests_cv.notify()

        while not self.is_window_locked():
            time.sleep(1e-4)
        return self.locked_window

    def free_window(self):
        with self._lock:
            self.status['window_locked'] = False
            self.locked_window.clear()

    def is_window_locked(self):
        return self.status['window_locked']

    def wait_until_empty_queue(self):
        while not self._queue.empty():
            time.sleep(1e-4)

    def interrupt_ba(self):
        self.optimizer.abort()

#------------------------------------------------------------------------
# a very simple implementation
class LoopDetection(object):
    def __init__(self, params):
        self.params = params
        self.nns = NearestNeighbors()

    def add_keyframe(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        self.nns.add_item(embedding, keyframe)

    def detect(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        kfs, ds = self.nns.search(embedding, k=20)

        if len(kfs) > 0 and kfs[0] == keyframe:
            kfs, ds = kfs[1:], ds[1:]
        if len(kfs) == 0:
            return None

        min_d = np.min(ds)
        for kf, d in zip(kfs, ds):
            if abs(kf.id - keyframe.id) < self.params.lc_min_inbetween_frames:
                continue
            if (np.linalg.norm(kf.position - keyframe.position) >
                    self.params.lc_max_inbetween_distance):
                break
            if d > self.params.lc_embedding_distance or d > min_d * 1.5:
                break
            return kf
        return None

class LoopClosing(object):
    def __init__(self, system, params):
        self.system = system
        self.params = params

        self.loop_detector = LoopDetection(params)
        self.optimizer = PoseGraphOptimization()

        self.loops = []
        self.stopped = False

        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def stop(self):
        self.stopped = True
        self._queue.put(None)
        self.maintenance_thread.join()
        print('loop closing stopped')

    def add_keyframe(self, keyframe):
        self._queue.put(keyframe)
        self.loop_detector.add_keyframe(keyframe)

    def add_keyframes(self, keyframes):
        for kf in keyframes:
            self.add_keyframe(kf)

    def maintenance(self):
        last_query_keyframe = None
        while not self.stopped:
            keyframe = self._queue.get()
            if keyframe is None or self.stopped:
                return

            # check if this keyframe share many mappoints with a loop keyframe
            covisible = sorted(
                keyframe.covisibility_keyframes().items(),
                key=lambda _: _[1], reverse=True)
            if any([(keyframe.id - _[0].id) > 5 for _ in covisible[:2]]):
                continue

            if (last_query_keyframe is not None and
                    abs(last_query_keyframe.id - keyframe.id) < 3):
                continue

            detected = self.loop_detector.detect(keyframe)
            if detected is None:
                continue

            query_keyframe = keyframe
            match_keyframe = detected

            result = match_and_estimate(
                query_keyframe, match_keyframe, self.params)

            if result is None:
                continue
            if (result.n_inliers < max(self.params.lc_inliers_threshold,
                                       result.n_matches * self.params.lc_inliers_ratio)):
                continue

            dist = result.correction.position()
            if self.params.ground:
                dist = dist[:2]
            if np.abs(dist).max() > self.params.lc_distance_threshold:
                continue

            self.loops.append(
                (match_keyframe, query_keyframe, result.constraint))
            query_keyframe.set_loop(match_keyframe, result.constraint)

            # We have to ensure that the mapping thread is on a safe part of code,
            # before the selection of KFs to optimize
            safe_window = self.system.mapping.lock_window()
            safe_window.add(self.system.reference)
            for kf in self.system.reference.covisibility_keyframes():
                safe_window.add(kf)

            # The safe window established between the Local Mapping must be
            # inside the considered KFs.
            considered_keyframes = self.system.graph.keyframes()

            self.optimizer.set_data(considered_keyframes, self.loops)

            before_lc = [
                g2o.Isometry3d(kf.orientation, kf.position) for kf in safe_window]

            # Propagate initial estimate through 10% of total keyframes
            # (or at least 20 keyframes)
            d = max(20, len(considered_keyframes) * 0.1)
            propagator = SmoothEstimatePropagator(self.optimizer, d)
            propagator.propagate(self.optimizer.vertex(match_keyframe.id))

            # self.optimizer.set_verbose(True)
            self.optimizer.optimize(20)

            # Exclude KFs that may being use by the local BA.
            self.optimizer.update_poses_and_points(
                considered_keyframes, exclude=safe_window)

            self.system.stop_adding_keyframes()

            # Wait until mapper flushes everything to the map
            self.system.mapping.wait_until_empty_queue()
            while self.system.mapping.is_processing():
                time.sleep(1e-4)

            # Calculating optimization introduced by local mapping while loop was been closed
            for i, kf in enumerate(safe_window):
                after_lc = g2o.Isometry3d(kf.orientation, kf.position)
                corr = before_lc[i].inverse() * after_lc

                vertex = self.optimizer.vertex(kf.id)
                vertex.set_estimate(vertex.estimate() * corr)

            self.system.pause()

            for keyframe in considered_keyframes[::-1]:
                if keyframe in safe_window:
                    reference = keyframe
                    break
            uncorrected = g2o.Isometry3d(
                reference.orientation,
                reference.position)
            corrected = self.optimizer.vertex(reference.id).estimate()
            T = uncorrected.inverse() * corrected  # close to result.correction

            # We need to wait for the end of the current frame tracking and ensure that we
            # won't interfere with the tracker.
            while self.system.is_tracking():
                time.sleep(1e-4)
            self.system.set_loop_correction(T)

            # Updating keyframes and map points on the lba zone
            self.optimizer.update_poses_and_points(safe_window)

            # keyframes after loop closing
            keyframes = self.system.graph.keyframes()
            if len(keyframes) > len(considered_keyframes):
                self.optimizer.update_poses_and_points(
                    keyframes[len(considered_keyframes) - len(keyframes):],
                    correction=T)

            for m13, _ in result.stereo_matches:
                query_meas = result.query_stereo_measurements[m13.queryIdx]
                match_meas = result.match_stereo_measurements[m13.trainIdx]

                new_query_meas = Measurement(
                    Measurement.Type.STEREO,
                    Measurement.Source.REFIND,
                    query_meas.get_keypoints(),
                    query_meas.get_descriptors())
                self.system.graph.add_measurement(
                    query_keyframe, match_meas.mappoint, new_query_meas)

                new_match_meas = Measurement(
                    Measurement.Type.STEREO,
                    Measurement.Source.REFIND,
                    match_meas.get_keypoints(),
                    match_meas.get_descriptors())
                self.system.graph.add_measurement(
                    match_keyframe, query_meas.mappoint, new_match_meas)

            self.system.mapping.free_window()
            self.system.resume_adding_keyframes()
            self.system.unpause()

            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    return
            last_query_keyframe = query_keyframe

def match_and_estimate(query_keyframe, match_keyframe, params):
    query = defaultdict(list)
    for m in query_keyframe.measurements():
        if m.from_triangulation():
            query['measurements'].append(m)
            query['kps1'].append(m.get_keypoint(0))
            query['kps2'].append(m.get_keypoint(1))
            query['desps1'].append(m.get_descriptor(0))
            query['desps2'].append(m.get_descriptor(1))
            n = len(query['matches'])
            query['matches'].append(cv2.DMatch(n, n, 0))

    match = defaultdict(list)
    for m in match_keyframe.measurements():
        if m.from_triangulation():
            match['measurements'].append(m)
            match['kps1'].append(m.get_keypoint(0))
            match['kps2'].append(m.get_keypoint(1))
            match['desps1'].append(m.get_descriptor(0))
            match['desps2'].append(m.get_descriptor(1))
            n = len(match['matches'])
            match['matches'].append(cv2.DMatch(n, n, 0))

    stereo_matches = query_keyframe.feature.circular_stereo_match(
        query['desps1'], query['desps2'], query['matches'],
        match['desps1'], match['desps2'], match['matches'],
        params.matching_distance,
        params.lc_inliers_threshold)

    n_matches = len(stereo_matches)
    if n_matches < params.lc_inliers_threshold:
        return None

    for m13, _ in stereo_matches:
        i, j = m13.queryIdx, m13.trainIdx
        query['px'].append(query['kps1'][i].pt)
        query['pt'].append(query['measurements'][i].view)
        match['px'].append(match['kps1'][j].pt)
        match['pt'].append(match['measurements'][j].view)

    # query_keyframe's pose in match_keyframe's coordinates frame
    T13, inliers13 = solve_pnp_ransac(
        query['pt'], match['px'], match_keyframe.cam.intrinsic)

    T31, inliers31 = solve_pnp_ransac(
        match['pt'], query['px'], query_keyframe.cam.intrinsic)

    if T13 is None or T13 is None:
        return None

    delta = T31 * T13
    if (g2o.AngleAxis(delta.rotation()).angle() > 0.1 or
            np.linalg.norm(delta.translation()) > 0.5):  # 5.7° or 0.5m
        return None

    n_inliers = len(set(inliers13) & set(inliers31))
    query_pose = g2o.Isometry3d(
        query_keyframe.orientation, query_keyframe.position)
    match_pose = g2o.Isometry3d(
        match_keyframe.orientation, match_keyframe.position)
    # TODO: combine T13 and T31
    constraint = T13
    estimated_pose = match_pose * constraint
    correction = query_pose.inverse() * estimated_pose

    return namedtuple('MatchEstimateResult',
                      ['estimated_pose', 'constraint', 'correction', 'query_stereo_measurements',
                       'match_stereo_measurements', 'stereo_matches', 'n_matches', 'n_inliers'])(
        estimated_pose, constraint, correction, query['measurements'],
        match['measurements'], stereo_matches, n_matches, n_inliers)

def solve_pnp_ransac(pts3d, pts, intrinsic_matrix):
    val, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(pts3d), np.array(pts),
        intrinsic_matrix, None, None, None,
        False, 50, 2.0, 0.99, None)
    if inliers is None or len(inliers) < 5:
        return None, None

    T = g2o.Isometry3d(cv2.Rodrigues(rvec)[0], tvec)
    return T, inliers.ravel()

class NearestNeighbors(object):
    def __init__(self, dim=None):
        self.n = 0
        self.dim = dim
        self.items = dict()
        self.data = []
        if dim is not None:
            self.data = np.zeros((1000, dim), dtype='float32')

    def add_item(self, vector, item):
        assert vector.ndim == 1
        if self.n >= len(self.data):
            if self.dim is None:
                self.dim = len(vector)
                self.data = np.zeros((1000, self.dim), dtype='float32')
            else:
                self.data.resize(
                    (2 * len(self.data), self.dim), refcheck=False)
        self.items[self.n] = item
        self.data[self.n] = vector
        self.n += 1

    def search(self, query, k):  # searching from 100000 items consume 30ms
        if len(self.data) == 0:
            return [], []

        ds = np.linalg.norm(query[np.newaxis, :] - self.data[:self.n], axis=1)
        ns = np.argsort(ds)[:k]
        return [self.items[n] for n in ns], ds[ns]
#------------------------------------------------------------------------

