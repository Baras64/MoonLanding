import sys
sys.path.append('..')

from simple_pid import PID
import numpy as np
from matplotlib import pyplot as plt
from pid.gridworld import GridSystem

if __name__ == "__main__":
    controlled_system = GridSystem(wind='random')

    Kp = 0.3
    Ki = 0.05
    Kd = 0.001
    pid = PID(Kp, Ki, Kd, setpoint=controlled_system.reference)

    # Assume we have a system we want to control in controlled_system
    s = controlled_system.update([0, 0])

    while True:
        # Compute new output from the PID according to the systems current value
        control = pid(s)
        controlled_system.render()
        
        # Feed the PID output to the system and get its current value
        s = controlled_system.update(control)


        error = np.sum(np.square(controlled_system.reference - controlled_system.state))
        if (np.abs(controlled_system.state_history - controlled_system.reference) < 0.1).all():
            controlled_system.reset()
            pid = PID(Kp, Ki, Kd, setpoint=controlled_system.reference)