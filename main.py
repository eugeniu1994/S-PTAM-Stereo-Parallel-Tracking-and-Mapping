import numpy as np
import time
import g2o
import cv2
from threading import Thread

from utils.params import Params
from utils.sptam import SPTAM
from utils.util import *
from utils.viewer import MapViewer

class Stereo_PTAM(object):
    def __init__(self, path, visualize = True):
        # data set path
        self.path = path
        self.visualize = visualize

        self.params = Params()  # parameters
        self.dataset = KITTI_dataset_reader(path)  # data set reader KITTI format
        w, h, fx, fy, cx, cy = self.dataset.cam.width, self.dataset.cam.height, self.dataset.cam.fx, self.dataset.cam.fy, self.dataset.cam.cx, self.dataset.cam.cy
        self.camera = Camera(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h,
                        frustum_near=self.params.frustum_near, frustum_far=self.params.frustum_far,
                        baseline=self.dataset.cam.baseline)

        self.sptam = SPTAM(self.params)

    def run(self):
        if self.visualize:
            self.viewer = MapViewer(self.sptam, self.params)

        for i in range(len(self.dataset)):
            featurel = ImageFeature(self.dataset.left[i], self.params)
            featurer = ImageFeature(self.dataset.right[i], self.params)
            timestamp = self.dataset.timestamps[i]

            start = time.time()
            t = Thread(target=featurer.extract)
            t.start()
            featurel.extract()
            t.join()
            frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, self.camera, timestamp=timestamp)
            if not self.sptam.is_initialized():
                self.sptam.initialize(frame)
            else:
                self.sptam.track(frame)
            duration = time.time() - start
            print('duration', duration)
            print()

            if self.visualize:
                self.viewer.update()

        print('Done')
        self.sptam.stop()
        if self.visualize:
            self.viewer.stop()

if __name__ == '__main__':
    path = '/home/eugeniu/Desktop/KITTI/data_odometry_gray/dataset/sequences/05'

    slam_obj = Stereo_PTAM(path=path, visualize=True)
    slam_obj.run()

