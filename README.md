# Solution 1 V2.0
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
solution. 

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
- GPU: Quadro RTX4000
## Note
1. Generally, the performance of this solution is dependent on the frame rate of the csi camera. The \
recommended throttle (PWM) setting are as follows: 
- frame rate below 30Hz, throttle: up to 0.1 
- frame_rate 30 - 40Hz, throttle: up to 0.16    
- frame_rate 40 - 50Hz, throttle: up to 0.17    
- frame_rate above 50Hz, throttle: up to 0.18 
2. The qlab's fps will also influence the performance of this solution, the recommended fps for \
this solution is 30. if the fps is much lower than 30, the recommended throttle is 0.1
3. Sometimes not all processes are properly activated due to some communication issues \
with the qlab, you can rerun the solution after entering the map again.

