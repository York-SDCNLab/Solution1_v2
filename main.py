# python imports
import sys
import time
from multiprocessing import Process, Event, Manager
# modeule imports
from core.roadmap.acc_roadmap import ACCRoadMap 
# from plan_vision_linefollow.script import writer_tasks, car_tasks
from plan_vision_linefollow import start_car, run_control_process, run_observer_process
from plan_vision_linefollow.utils import EventWrapper
from scripts.Setup_Competition import setup
# from collection_demo import *

def spawn_on_node(node_id=24) -> None: 
    """
    Spawn the car on the specified node (0 - 24). 

    Parameters:
        node_id (int): The node id to spawn the car on.
    """
    # init the roadmap
    roadmap: ACCRoadMap = ACCRoadMap() 
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    setup(initialPosition=[x_pos, y_pose, 0], initialOrientation=[0, 0, angle]) 

def start_multiprocess_car(node_id: int = 24, throttle: float = 0.1) -> None:
    """
    Start the car with the vision line follow policy and the detection process.

    Parameters:
        node_id (int): The node id to spawn the car on.
        throttle (float): The throttle value for the car.
    """
    spawn_on_node(node_id=node_id) # spawn the car on the specific node
    manager = Manager()
    events = ["stop_sign", "horizontal_line", "red_light", "unknown_error"]
    events_warpper: EventWrapper = EventWrapper(manager)
    events_warpper.setup(events)
    activate_event = Event()
    # initialize processes
    observer_process: Process = Process(
        target=run_observer_process, 
        args=(events_warpper, activate_event)
    )
    # start the observer process
    observer_process.start()
    while not activate_event.is_set():
        time.sleep(0.01)
    time.sleep(2) # wait for the observer to start
    run_control_process(events_warpper, throttle=throttle)    
    try: 
        while True: 
            time.sleep(100)
    except KeyboardInterrupt: 
        observer_process.terminate()
    except Exception as e:
        print(e)
    finally:
        print("Execution complete")
        sys.exit(0)

if __name__ == "__main__":
    # Uncomment the spawn on node and traffic light process if you want
    if len(sys.argv) == 1: 
        # Recommended throttle value is 0.13 - 0.18
        start_multiprocess_car(node_id=24, throttle=0.18) 
    elif len(sys.argv) > 1:
        try: 
            throttle: float = float(sys.argv[1])
            start_multiprocess_car(node_id=24, throttle=throttle)
        except Exception: 
            print("The input should be a float") 
    else: 
        print("The input should be a float")