import cv2

class Params(object):
    def __init__(self, config='GFTT-BRIEF'):

        self.pnp_min_measurements = 10  # the number of measurements for PnP algorithm
        self.pnp_max_iterations = 10  # the number of iterations for PnP
        self.init_min_points = 7# 10

        self.local_window_size = 10  # local window for bundle adjustment
        self.ba_max_iterations = 10  # number of iterations for BA

        self.min_tracked_points_ratio = 0.5

        # loop closure
        self.lc_min_inbetween_frames = 10  # frames
        self.lc_inliers_threshold = 15
        self.lc_inliers_ratio = 0.5
        self.lc_max_iterations = 20

        self.view_camera_size = 1

        if config == "GFTT-BRIEF":
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0,
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'GFTT-BRISK':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=2000, minDistance=15.0,
                qualityLevel=0.01, useHarrisDetector=False)

            self.descriptor_extractor = cv2.BRISK_create()

        elif config == 'ORB-ORB':
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
            self.descriptor_extractor = self.feature_detector

        #change here the descriptor -> add another options
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # FLANN parameters
        #FLANN_INDEX_KDTREE = 1
        #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #search_params = dict(checks=50)  # or pass empty dictionary
        #self.descriptor_matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.matching_cell_size = 15  # pixels
        self.matching_neighborhood = 3
        self.matching_distance = 30

        self.frustum_near = 0.1  # meters
        self.frustum_far = 1000.0

        self.ground = True

        self.lc_max_inbetween_distance = 50 #meters
        self.lc_distance_threshold = 15  #meters
        self.lc_embedding_distance = 20.0

        self.view_image_width =560# 400
        self.view_image_height =220# 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500  # -10
        self.view_viewpoint_z = -100  # -0.1
        self.view_viewpoint_f = 2000













