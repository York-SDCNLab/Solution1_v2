# Solution 1  V2.0
## Introduction
This is a solution for a competition. The control system adopts an event driven \
architecture and features two asynchronous processes for control and monitoring, \
respectively. 
## Usage
1. To run the solution, first clone this repository: \
```git clone https://github.com/York-SDCNLab/Solution1_v2.git```

2. Navigate to /path/to/Solution1 and run ```pip install -e .``` to setup \
the local dependencies. Once done, you can install the project dependencies by runnig \
```pip install -r requirements.txt```

3. Now the project is set up! You can run ```python main.py``` to begin running our \
solution. Note that we have separate processes in ```main.py``` for running the \
traffic light control and map generation. These are identical to the provided \
```Setup_Competition.py``` and ```Traffic_Lights_competition.py``` scripts, however \
if you want to run these scripts manually, you can comment the following lines in main.py.

```
spawn_on_node(node_id=node_id) # spawn the car on the specific node
```

```
traffic_light_process = Process(
    target=run_traffic_light, 
    args=('auto', 5, None))
activate_event.clear()
traffic_light_process.start()
time.sleep(2) # wait for the traffic light process to start
```
## Recommended Hardware Requirements
- CPU: Intel Xeno W-1290P 
- RAM: 16GB 
- GPU: GTX1650
## Note
1. Generally, the performance of this solution is dependent on the frame rate of the csi camera. The \
recommended throttle (PWM) setting are as follows: 
- frame rate below 30Hz, throttle: up to 0.1 
- frame_rate 30 - 40Hz, throttle: up to 0.16    
- frame_rate 40 - 50Hz, throttle: up to 0.17    
- frame_rate above 50Hz, throttle: up to 0.18 
2. Sometimes not all processes are properly activated due to some communication issues \
with the qlab, you can rerun the solution after entering the map again.

