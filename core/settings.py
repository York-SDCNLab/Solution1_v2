import numpy as np 

# policy parameter settings
DEFAULT_SLOPE_OFFSET: float = -0.7718322998996283
DEFAULT_INTERCEPT_OFFSET: float = 400.0
EDGES_LOWER_BOUND: int = 50
EDGES_UPPER_BOUND: int = 150
HOUGH_ANGLE_UPPER_BOUND: int = 160
HOUGH_ANGLE_LOWER_BOUND: int = 100
HOUGH_CONFIDENT_THRESHOLD: int = 85
THRESH_UPPER_BOUND: int = 255
THRESH_LOWER_BOUND: int = 100
DEFAULT_K_P: float = -1.0163443448
DEFAULT_K_I: float = -0.000
DEFAULT_K_D: float = -0.19878977558

# camera parameter settings
CSI_CAMERA_SETTING: dict = {
    'focal_length': np.array([[157.9], [161.7]], dtype = np.float64) , 
    'principle_point': np.array([[168.5], [123.6]], dtype = np.float64), 
    'position': np.array([[0], [0], [0.14]], dtype = np.float64), 
    'orientation': np.array([[ 0, 0, 1], [ 1, 0, 0], [ 0, -1, 0]], dtype = np.float64),
    'frame_width': 820,
    'frame_height': 410,
    'frame_rate': 70.0
}

RGBD_CAMERA_SETTING: dict = {
    'mode': 'RGB', 
    'frame_width_rgb': 640, 
    'frame_height_rgb': 480, 
    'frame_rate_rgb': 30.0, 
    'frame_width_depth': 640, 
    'frame_height_depth': 480, 
    'frame_rate_depth': 15.0, 
    'device_id': '0@tcpip://localhost:18965'
}