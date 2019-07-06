"""
This script uses BFS to find the optimal path for kinematic bicycle model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import RK45

class VehiclePlanner:
    
    def __init__(self):
        # Boundary of feasible space
        self.width = self.height = 100
        self.verCorridor = 20
        self.horCorridor = 20

        # Initial Position of vehicle
        self.x0 = self.width - self.verCorridor / 2.0
        self.y0 = 0.0
        self.theta0 = np.pi/2

        # Car parameters
        self.v = 1
        self.u = [-40/180*np.pi, 0.0, 40/180*np.pi]  # Steering angle: turn left, straight, turn right
        self.lf = self.lr = 2
        self.beta = np.arctan(0.5*np.tan(40/180*np.pi))
        self.dt = np.pi/2/(self.v*np.sin(self.beta)/self.lr)  # time step for integration. vehicle +- 90 degrees.
        self.r = 2.5  # The car is simplified as a sphere with radius = 2.5 in collision detection.
        
        #Final Position
        self.xf = self.verCorridor / 2.0
        self.yf = 0.0
        self.thetaf = -np.pi/2

        # Plot
        self.ax = self.plotBoundary()

    def plotBoundary(self):
        """
        This function plots the boundary and obstacles in world space. This should only be called once.
        """
        plt.figure(1, figsize=(12, 12))
        # Outer boundary
        plt.plot([0+self.verCorridor, self.width-self.verCorridor], [0, 0], 'r', linewidth=4.0)
        plt.plot([self.width, self.width], [0, self.height], 'r', linewidth=4.0)
        plt.plot([self.width-self.verCorridor, self.verCorridor], [self.height, self.height], 'r', linewidth=4.0)
        plt.plot([0, 0], [self.height, 0], 'r', linewidth=4.0)

        # Obstacles
        plt.plot([self.verCorridor, self.verCorridor], [0, (self.height-self.horCorridor)/2], 'r', linewidth=4.0)
        plt.plot([self.verCorridor, self.verCorridor], [(self.height+self.horCorridor)/2, self.height], 'r', linewidth=4.0)
        plt.plot([self.width-self.verCorridor, self.width-self.verCorridor], [0, (self.height-self.horCorridor)/2], 'r', linewidth=4.0)
        plt.plot([self.width-self.verCorridor, self.width-self.verCorridor], [(self.height+self.horCorridor)/2, self.height], 'r', linewidth=4.0)
        plt.plot([self.verCorridor, self.width-self.verCorridor], [(self.height-self.horCorridor)/2, (self.height-self.horCorridor)/2], 'r', linewidth=4.0)
        plt.plot([self.verCorridor, self.width-self.verCorridor], [(self.height+self.horCorridor)/2, (self.height+self.horCorridor)/2], 'r', linewidth=4.0)
        
        # Add initial and goal
        ax = plt.gca()
        ax.add_patch(Rectangle((self.verCorridor/2-1, 0-2), 2, 4, color='green'))
        ax.add_patch(Rectangle((self.width-self.verCorridor/2-1, 0-2), 2, 4, color='yellow'))

        # Add the initial configuration to the plot
        plt.plot([self.width-self.verCorridor/2], [0], 'bo', markersize=10.0)

        plt.xlim(-10, 110)
        plt.ylim(-10, 110)
        plt.draw()
        plt.pause(1e-3)

        return ax

    def turnLeft(self, t, y):
        """
        Bicycle model for vehicles that turns left. It returns dy/dt and is the callable used in RK45.
        y is the state variable [x, y, theta]. The velocity doesn't change.
        """
        beta = self.beta
        
        dx_dt = self.v*np.cos(y[2]+beta)
        dy_dt = self.v*np.sin(y[2]+beta)
        dtheta_dt = self.v/self.lr*np.sin(beta)

        return [dx_dt, dy_dt, dtheta_dt]

    def turnRight(self, t, y):
        """
        Bicycle model for vehicles that turns right. It returns dy/dt and is the callable used in RK45.
        y is the state variable [x, y, theta]. The velocity doesn't change.
        """
        beta = -self.beta
        
        dx_dt = self.v*np.cos(y[2]+beta)
        dy_dt = self.v*np.sin(y[2]+beta)
        dtheta_dt = self.v/self.lr*np.sin(beta)

        return [dx_dt, dy_dt, dtheta_dt]
    
    def keepStraight(self, t, y):
        """
        Bicycle model for vehicles that keeps straight. It returns dy/dt and is the callable used in RK45.
        y is the state variable [x, y, theta]. The velocity doesn't change.        
        """
        beta = 0.0
        
        dx_dt = self.v*np.cos(y[2]+beta)
        dy_dt = self.v*np.sin(y[2]+beta)
        dtheta_dt = self.v/self.lr*np.sin(beta)

        return [dx_dt, dy_dt, dtheta_dt]

    def simulate(self, x, y, theta, oper="left"):
        """
        Run one step of simulation to get the new configuration. It should return the history of x, y and theta in a list.
        """
        states = [x, y, theta]
        
        if oper == "left":
            dx_dt = self.turnLeft
        elif oper == "right":
            dx_dt = self.turnRight
        elif oper == "straight":
            dx_dt = self.keepStraight
        else:
            raise RuntimeError("Unsupported operation type ", oper)

        integrator = RK45(fun=dx_dt, t0=0, y0=states, t_bound=self.dt, max_step=self.dt/20)
        result = [states]
        print ("="*50)
        print ("Operation = ", oper)
        print ("t = 0.0, states=", states)
        while integrator.status == "running":
            integrator.step()
            states = [integrator.y[0], integrator.y[1], integrator.y[2]]
            result.append(states)
            print ("t = {}, states=".format(integrator.t), states)
            if self.collide(states[0], states[1]):
                integrator.status = "failed"

        if integrator.status == "failed":
            raise RuntimeError("Simulation fails or collision detected.")
        else:
            return result  # The last list is the final configuration in W space.

    def collide(self, x, y):
        """
        Return true if collision is detected at the given point.
        """
        # Check the left outer boundary
        if x <= 0 + self.r:
            return True
        
        # Check the right outer boundary
        if x >= self.width - self.r:
            return True
        
        # Check the lower middle obstacle
        if x >= self.verCorridor - self.r and x <= self.width - self.verCorridor + self.r and y <= (self.height-self.horCorridor)/2 + self.r:
            return True
        
        # Check the upper middle obstacle
        if x >= self.verCorridor - self.r and x <= self.width - self.verCorridor + self.r and y >= (self.height+self.horCorridor)/2 - self.r:
            return True
        
        return False

    

if __name__ == "__main__":
    v_planner = VehiclePlanner()
    print ("Initialization complete")
    v_planner.simulate(90, 0, np.pi/2, "left")
    v_planner.simulate(90, 0, np.pi/2, "right")
    v_planner.simulate(95, 0, np.pi/2, "straight")
