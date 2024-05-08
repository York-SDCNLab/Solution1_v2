# python imports
import time
import math
# 3rd party imports
import cv2
import numpy as np 
# import torch
# quanser imports
from pal.products.qcar import QCar
# custom imports
from core.sensor import VirtualCSICamera, VirtualRGBDCamera
from core.settings import DEFAULT_K_P, DEFAULT_K_I, DEFAULT_K_D
from .policy import VisualLineFollowing
from .exceptions import NoContourException, NoImageException, HaltException, StopException
from .decision_pipeline import DecisionMaker, Compose
from .utils import EventWrapper


class VLFCar: 
    """
    The Visual Line Following Car class for the QCar.

    Attributes:
        sample_time (float): sample_time for the QCar.
        running_gear (QCar): Running gear for the QCar.
        leds (np.ndarray): LED array for the QCar.
        front_csi (VirtualCSICamera): Front CSI camera for the QCar.

    Methods:
        setup(throttle: float = 0.1) -> None:
            Sets up the QCar with the specified throttle value.
        halt_car(steering: float = 0.0, halt_time: float = 1.0) -> None:
            Halts the QCar.
        execute() -> None:
            Executes the QCar operation.
    """

    def __init__(self) -> None:
        """
        Initializes the VLFCar class for the QCar.

        Returns:    
            None
        """
        # self.sample_time: float = 4 
        self.running_gear: QCar = QCar()
        self.leds: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.front_csi: VirtualCSICamera = VirtualCSICamera()
        
    def setup(self, throttle: float = 0.1) -> None:
        """
        Sets up the QCar with the specified throttle value.

        Parameters:
            throttle (float): Throttle value for the QCar.

        Returns:
            None
        """
        # self.diff: Calculus = Calculus().differentiator_variable(1 / self.sample_time)
        # _ = next(self.diff)
        # self.time_step: float = self.sample_time
        self.policy: VisualLineFollowing = VisualLineFollowing(throttle=throttle)
        self.policy.setup(k_p=DEFAULT_K_P, k_i=DEFAULT_K_I, k_d=DEFAULT_K_D) # can change K-i, K-p, K-d here
        self.brake_time: float = (throttle - 0.1) / 0.16
        self.start: float = time.time()

    def halt_car(self, steering: float = 0.0, halt_time: float = 1.0) -> None: 
        """
        Halts the QCar.

        Returns:
            None
        """
        self.running_gear.read_write_std(throttle=0, steering=steering, LEDs=self.leds)
        time.sleep(halt_time)
        self.policy.start = time.time()

    def execute(self) -> None: 
        """
        The QCar execution function.

        Returns:
            None
        """
        # start time for stop sign
        try: 
            # start = time.time()
            image: np.ndarray = self.front_csi.read_image()
            self.policy.execute_policy(image)
            steering: float = self.policy.steering
            throttle: float = self.policy.throttle * abs(math.cos(2.05 * steering))
            self.running_gear.read_write_std(throttle=throttle, steering=steering, LEDs=self.leds)
        except NoContourException:
            pass 
        except NoImageException:
            pass 


class EVLFControl(VLFCar): 
    """
    The Extended Visual Line Following Control class for the QCar control process.

    Attributes:
        event_wrapper (EventWrapper): Event wrapper class for the QCar receive event 
        from the observer process.
        throttle (float): Throttle value for the QCar.
        halt_steering (float): Halt steering value for the QCar.

    Methods:
        handle_events() -> None:
            Handles the events for the QCar control process.
        execute() -> None:
            Executes the QCar control process.
    """
    
    def __init__(self, event_wrapper: EventWrapper, throttle: float = 0.1) -> None:
        """
        Initializes the EVLFControl class for the QCar control process.

        Parameters:
            event_wrapper (EventWrapper): Event wrapper class for the QCar receive event 
            from the observer process.
            throttle (float): Throttle value for the QCar.

        Returns:
            None
        """
        super().__init__()
        self.event_wrapper: EventWrapper = event_wrapper
        self.reduce_factor: float = 1.0
        self.halt_steering: float = 0.0
    
    def handle_events(self) -> None: 
        """
        Handles the events for the QCar control process.

        Returns:
            None
        """
        if self.event_wrapper.event.is_set(): 
            shared_events: dict = self.event_wrapper.event_types
            if shared_events['horizontal_line']:
                self.reduce_factor = 0.08 / self.policy.throttle
            else: 
                self.reduce_factor = 1.0

            if shared_events['red_light']:
                self.halt_steering = self.policy.steering
                self.halt_car(steering=self.halt_steering, halt_time=0.1)
                raise HaltException(stop_time=0.1, message="Red Light")
            if shared_events['stop_sign']:
                stop_time = 3 + self.brake_time # random.randint(1, 3)
                self.halt_steering = 0.0
                self.halt_car(steering=self.halt_steering, halt_time=stop_time)
                raise HaltException(stop_time=3, message="Stop Sign")
        else: 
            self.reduce_factor = 1.0
            
    def execute(self) -> None:
        """
        Executes the QCar control process.

        Returns:
            None

        Raises:
            NoContourException: Raises an exception if no contour is detected in the image 
                processing operation.
            HaltException: Raises an exception if the car is required to halt.
            Exception: Raises an exception if an unknown error occurs during the execution.
        """
        try: 
            self.handle_events()
            csi_image: np.ndarray = self.front_csi.read_image() #self.front_csi.read_image()
            # delta_t = time.time() - self.start
            self.policy.execute_policy(origin_image=csi_image)
            self.start = time.time()
            steering: float = self.policy.steering
            throttle: float = self.policy.throttle * abs(math.cos(2.7 * steering)) * self.reduce_factor
            self.running_gear.read_write_std(throttle=throttle, steering=steering, LEDs=self.leds)
            # cv2.waitKey(1)
        except NoContourException:
            # self.policy.start = time.time()
            pass
        except NoImageException: 
            pass
        except HaltException as e:
            print(f"Stop {e.stop_time} seconds for {e.message}")
        except Exception as e:
            raise e # transmit the exception to the main process


class EVLFObserver: 
    """
    The Extended Observer class for the QCar observer process, continuously observing the Objects
    with RGB camera.

    Attributes:
        events (EventWrapper): Event wrapper class for the QCar receive event from the observer process.
        rgbd (VirtualRGBDCamera): Virtual RGBD camera for the QCar observer process.
        pipeline (DecisionMaker): Decision maker for the QCar observer process.

    Methods:
        execute() -> None:
            Executes the QCar observer process.
    """
    def __init__(self, events: EventWrapper, file_path: str) -> None:
        """
        Parameters:
            events (EventWrapper): Event wrapper class for the QCar receive event from the observer process.
            file_path (str): Model file path for the QCar observer process.
        
        Returns:
            None
        """
        self.events: EventWrapper = events
        self.rgbd: VirtualRGBDCamera = VirtualRGBDCamera()
        self.pipeline = DecisionMaker(
            classic_traffic_pipeline=True,
            network_class=None, 
            input_preprocess= None,
            output_postprocess=lambda x: x.argmax().item(),
            weights_file=file_path, # 'plan_vision_linefollow/model_weights_final_1999.qcar',
            device='cuda'
        )

    def execute(self) -> None: 
        """
        The QCar observer process execution function.

        Returns:
            None

        Raises:
            Exception: Raises an exception if an unknown error occurs during the execution.
        """
        try: 
            rgbd_image: np.ndarray = self.rgbd.read_rgb_image()
            if rgbd_image is not None: 
                self.pipeline(rgbd_image)
                detection_flags: dict = self.pipeline.detection_flags
                for key, value in detection_flags.items():
                    if value: 
                        self.events.set(key)
                    else: 
                        self.events.clear(key)
                # cv2.waitKey(1)
        except Exception as e:
            print(e)
            raise e