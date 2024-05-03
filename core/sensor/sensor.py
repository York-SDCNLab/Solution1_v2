import time 
import math 
import numpy as np 

from pal.utilities.vision import Camera2D
from pal.utilities.vision import Camera3D 
from pal.products.qcar import QCarGPS 

from core.template import ServiceModule
from core.settings import CSI_CAMERA_SETTING
from core.settings import RGBD_CAMERA_SETTING


class VirtualCSICamera(ServiceModule): # wrapper class, implement more functions if needed 
    def __init__(self, id=3) -> None: 
        self.id = id 
        self.camera: Camera2D = Camera2D(
            cameraId=str(id) + "@tcpip://localhost:"  + str(18961+id), 
            frameWidth=CSI_CAMERA_SETTING['frame_width'],
            frameHeight=CSI_CAMERA_SETTING['frame_height'],
            frameRate=CSI_CAMERA_SETTING['frame_rate'], 
            focalLength=CSI_CAMERA_SETTING['focal_length'], 
            principlePoint=CSI_CAMERA_SETTING['principle_point'], 
            position=CSI_CAMERA_SETTING['position'], 
            orientation=CSI_CAMERA_SETTING['orientation']
        )

    def terminate(self) -> None: 
        self.camera.terminate() 

    def read_image(self) -> None: # image without any modification 
        if self.camera.read(): 
            return self.camera.imageData
        return None


class VirtualRGBDCamera(ServiceModule): 
    def __init__(self) -> None: 
        self.camera: Camera3D = Camera3D(
            mode=RGBD_CAMERA_SETTING['mode'], 
            frameWidthRGB=RGBD_CAMERA_SETTING['frame_width_rgb'], 
            frameHeightRGB=RGBD_CAMERA_SETTING['frame_height_rgb'], 
            frameRateRGB=RGBD_CAMERA_SETTING['frame_rate_rgb'], 
            frameWidthDepth=RGBD_CAMERA_SETTING['frame_width_depth'], 
            frameHeightDepth=RGBD_CAMERA_SETTING['frame_height_depth'], 
			frameRateDepth=RGBD_CAMERA_SETTING['frame_rate_depth'], 
            deviceId=RGBD_CAMERA_SETTING['device_id']
        )

    def terminate(self) -> None: 
        self.camera.terminate() 

    def read_rgb_image(self) -> np.ndarray: 
        if self.camera.read_RGB() != -1: 
            # cv2.imshow('RGBD Image', self.camera.imageBufferRGB)
            return self.camera.imageBufferRGB 

    def read_depth_image(self, data_mode:str = 'PX') -> np.ndarray: 
        if self.camera.read_depth(data_mode) != -1: 
            if data_mode == 'PX': 
                # cv2.imshow('RGBD PX', self.camera.imageBufferDepthPX)
                return self.camera.imageBufferDepthPX
            if data_mode == 'M': 
                # cv2.imshow('RGBD M', self.camera.imageBufferDepthM)
                return self.camera.imageBufferDepthM


class VirtualLidar(ServiceModule): 
    def __init__(self) -> None: 
        pass # will start implementation after the error -15 fixed 

class VirtualGPS(ServiceModule): 
    def __init__(self) -> None: 
        self.gps: QCarGPS = QCarGPS() 
        self.speed_vector: tuple = None
        self.speed_history: list = [] 

    def terminate(self) -> None: 
        self.gps.terminate() 
        # plot_line_chart(self.speed_history[1:], 'time', 'speed', 'speed chart') 

    def get_gps_state(self) -> tuple:  
        position_x: float = self.gps.position[0] 
        position_y: float = self.gps.position[1]
        orientation: float = self.gps.orientation[2] 

        return position_x, position_y, orientation 
    
    def calcualte_speed_vector(self, current_state, delta_t) -> tuple: 
        delta_x_sq: float = math.pow((current_state[0] - self.last_state[0]), 2)
        delta_y_sq: float = math.pow((current_state[1] - self.last_state[1]), 2)

        linear_speed: float = math.pow((delta_x_sq + delta_y_sq), 0.5) / delta_t 
        angular_speed: float = (current_state[2] - self.last_state[2]) / delta_t

        return linear_speed, angular_speed 
    
    def setup(self) -> None: 
        # create or overwrite the log 
        open("output/gps_log.txt", "w") 
        # init states 
        self.time_stamp: float = time.time() 
        self.gps.readGPS() # read gps info
        self.last_state: tuple = self.get_gps_state() 

    def read_gps_state(self) -> None:  
        # read current position 
        self.gps.readGPS()
        current_time: float = time.time() 
        self.current_state: tuple = self.get_gps_state() 
        # calculate absolute speed 
        if self.current_state != self.last_state or current_time - self.time_stamp >= 0.25: 
            delta_t: float = current_time - self.time_stamp 
            self.speed_vector = self.calcualte_speed_vector(self.current_state, delta_t)
            # self.speed_history.append(speed_vecotr[0]) 

            # os.system("cls")  
            # print(f"delta_t: {delta_t:.4f}s") 
            # print(f"last_x: {self.last_state[0]:.2f}, last_y: {self.last_state[1]:.2f},  last_orientation: {((180 / np.pi) * self.last_state[2]):.2f}°")    
            # print(f"x: {self.current_state[0]:.2f}, y: {self.current_state[1]:.2f},  orientation: {((180 / np.pi) * self.current_state[2]):.2f}°") 
            # print(f"speed: {speed_vecotr[0]:.4f} m/s, angular speed: {speed_vecotr[1]:.4f} rad/s") 
            
            # update time stamp  
            self.time_stamp = current_time 
            # update position 
            self.last_state = self.current_state