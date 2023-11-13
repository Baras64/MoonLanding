from agents import BaseAgent
import numpy as np

class OptimalControlAgent(BaseAgent):
    def __init__(self):
        self.kp_alt = +4.60245202  # proportional altitude
        self.kd_alt = -3.57703678  # derivative altitude
        self.kp_ang = -60.1764885  # proportional angle
        self.kd_ang = +40.0989124  # derivative angle

    def action(self, state):
        """ calculates settings based on pid control """
        x, y, vx, vy, theta, vtheta, left_leg, right_leg = state[:8]
        
        # Calculate setpoints (target values)
        alt_tgt = np.abs(x)
        ang_tgt = (.25*np.pi)*(x+vx)

        # Calculate error values
        alt_error = (alt_tgt - y)
        ang_error = (ang_tgt - theta)
        
        # Use PID to get adjustments
        alt_adj = self.kp_alt*alt_error + self.kd_alt*vy
        ang_adj = self.kp_ang*ang_error + self.kd_ang*vtheta
            
        # Gym wants them as np array (-1,1)
        a = np.array([alt_adj, ang_adj])
        a = np.clip(a, -1, +1)
        
        # If the legs are on the ground we made it, kill engines
        if(left_leg or right_leg):
            a[:] = 0   
        return a
