# This script solves the path planning problem for
# a planar double link pendulum with three obstacles.
# The pendulum is fixed to the ground and both joints(q1, q2)
# are free to move. Given the initial and final configuration, 
# the solver will return q1-t and q2-t.

import numpy as np
import matplotlib.pyplot as pyplot

class Node:
    """
    Node in configuration space (q1, q2)
    """
    def __init__(self, q, parent=None):
        assert (len(q) == 2)
        self.q = q
        self.parent = None

class Rectangle:
    """
    Robot Geometry. The robot contains multiple
    rectangles connected by revolute joint.
    """
    def __init__(self, l, w):
        assert (l>0)
        assert (w>0)
        self.l = l
        self.w = w
        self.x = 0
        self.y = 0
        self.theta = 0

    def setPosition(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def mesh(self):
        """
        Generate the vertex of mesh
        """
        x = np.linspace(-self.l/2.0, self.l/2.0, 50)
        y = np.linspace(-self.w/2.0, self.w/2.0, 5)
        x, y = np.meshgrid(x, y, indexing='ij')

        vertex = []
        for i in range(50):
            for j in range(5):
                vertex.append(self.move(x[i][j], y[i][j]))

        return vertex

    def move(self, x, y):
        x_new = np.cos(self.theta)*x - np.sin(self.theta)*y + self.x
        y_new = np.sin(self.theta)*x + np.cos(self.theta)*y + self.y
        return [x_new, y_new]

class Circle:
    """
    Obstacle Geometry.
    """
    def __init__(self, r, pos):
        assert (r>0)
        assert (len(pos))
        self.r = r
        self.pos = pos

    def inCircle(self, pos):
        assert (len(pos) == 2)
        disp = np.array(
            [pos[0]-self.pos[0], pos[1]-self.pos[1]]
        )

        dist = np.linalg.norm(disp)
        if dist-self.r > 1e-3:
            return False
        else:
            return True

class DoublePendulum:
    """
    Robot class.
    """
    def __init__(self, q0, qt, geometry):
        assert (len(q0) == 2)
        assert (len(qt) == 2)
        assert (len(geometry) == 2)

        self.q0 = q0
        self.qt = qt
        self.geo = geometry
    
    def collide(self, q, obstacles):
        """
        Return true if collision happens
        """
        l1 = self.geo[0].l
        l2 = self.geo[1].l
        x1 = l1/2*np.cos(q[0])
        y1 = l1/2*np.sin(q[0])
        theta1 = q[0]
        x2 = l1*np.cos(q[0]) + l2/2*np.cos(q[0]+q[1])
        y2 = l1*np.sin(q[0]) + l2/2*np.sin(q[0]+q[1])
        theta2 = q[0] + q[1]

        # Check link1
        self.geo[0].setPosition(x1, y1, theta1)
        mesh1 = self.geo[0].mesh()
        for circle in obstacles:
            for point in mesh1:
                if circle.inCircle(point):
                    return True

        # Check link2
        self.geo[1].setPosition(x2, y2, theta2)
        mesh2 = self.geo[1].mesh()
        for circle in obstacles:
            for point in mesh2:
                if circle.inCircle(point):
                    return True
        
        return False

class Planner:
    """
    Path planner for double pendulum. 
    Rapidly exploring random tree (RRT) is used here with identification.
    """
    def __init__(self, robot, obstacles, step_size, goal_rate):
        self.robot = robot
        self.obstacles = obstacles

        assert (step_size > 0)
        assert (goal_rate > 0)
        self.step = step_size
        self.goal_rate = goal_rate

        self.nodes = [Node(self.robot.q0)]
        self.goal = self.robot.qt

    

if __name__ == "__main__":
    # rect = Rectangle(10, 1)
    # rect.setPosition(5,5,np.pi/4)
    # mesh = rect.mesh()
    # for point in mesh:
    #     pyplot.plot([point[0]], [point[1]], 'ro')
    # pyplot.show()
    
    rect1 = Rectangle(1, 0.1)
    rect2 = Rectangle(1, 0.1)
    robot = DoublePendulum([0,0], [1,1], [rect1, rect2])
    obs1 = Circle(0.5, [1, 1])
    obs2 = Circle(0.5, [1,-1])

    print ("Check if there is any collision")
    print (robot.collide([0,np.pi/2], [obs1, obs2]))
    print (robot.collide([0,np.pi/3], [obs1, obs2]))
    print (robot.collide([0,np.pi/6], [obs1, obs2]))