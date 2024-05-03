# python imports
import time

RECOMMENDED_THROTTLES = {
    29: 0.1,
    30: 0.16,
    40: 0.17,
    50: 0.18
}

def test_control_rate(control) -> float:
    """
    Test the control rate of the QCar control process.

    Parameters:
        control: The control process for the QCar.

    Returns:
        float: The average control rate of the QCar control process.
    """
    test_history: list = []

    try: 
        test_counter: float = 0.0
        control.policy.start = time.time()
        while test_counter < 10: 
            control.execute()
            if control.policy.dt != 0.0:
                test_counter += 1
                if test_counter != 1:
                    test_history.append(1 / control.policy.dt)
                control.policy.dt = 0.0
    except Exception as e: 
        print(e)
    
    return sum(test_history) / len(test_history)

def get_recommended_throttle(average_rate: float) -> float: 
    """
    Get the recommended throttle value for the QCar control process.

    Parameters:
        average_rate (float): The average control rate of the QCar control process.
    
    Returns:
        float: The recommended throttle value for the QCar control process.
    """
    proccesed_rate: int = round(average_rate / 10) * 10
    closest_rate = min(RECOMMENDED_THROTTLES.keys(), key=lambda k: abs(k - proccesed_rate))
    if closest_rate in RECOMMENDED_THROTTLES.keys(): 
        return RECOMMENDED_THROTTLES[closest_rate]
    else: 
        return 0.1

if __name__ == "__main__":
    throttle: float = get_recommended_throttle()
    print(f"Recommended throttle value is {throttle}")