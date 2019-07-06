# This script solves the path planning problem for
# a planar double link pendulum with three obstacles.
# The pendulum is fixed to the ground and both joints(q1, q2)
# are free to move. Given the initial and final configuration, 
# the solver will return q1-t and q2-t.

import numpy as np
import matplotlib.pyplot as pyplot
import random, csv

class Node:
    """
    Node in configuration space (q1, q2)
    """
    def __init__(self, q, parent=None):
        assert (len(q) == 2)
        self.q = q
        self.parent = parent

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
    def __init__(self, robot, obstacles, step_size, goal_rate, max_iter=1000):
        self.robot = robot
        self.obstacles = obstacles

        assert (step_size > 0)
        assert (goal_rate > 0)
        assert (goal_rate < 1)
        self.step = step_size
        self.goal_rate = goal_rate
        self.max_iter = max_iter

        self.nodes = [Node(self.robot.q0)]
        self.goal = self.robot.qt

        random.seed(10)

    def distance(self, q1, q2):
        """
        Compute the Eucledian Norm between two configurations. 
        Special treatment is needed on boundary due to identification (0 = 2pi)
        """
        d0 = min(abs(q2[0]-q1[0]), abs(q2[0]-q1[0]+2*np.pi), abs(q2[0]-q1[0]-2*np.pi))
        d1 = min(abs(q2[1]-q1[1]), abs(q2[1]-q1[1]+2*np.pi), abs(q2[1]-q1[1]-2*np.pi))
        return np.linalg.norm([d0, d1])

    def getClosestNode(self, q):
        """
        Get the node closest to the new node q
        """
        dist = np.inf
        closest = None

        for node in self.nodes:
            dist_new = self.distance(q, node.q)
            if dist_new < dist:
                closest = node
                dist = dist_new
        
        return closest

    def getDirection(self, q1, q2):
        """
        Get the direction of movement from q1 to q2
        A unit vector is returned.
        """
        direction = np.array(
            [self.diff1D(q1[0], q2[0]), self.diff1D(q1[1], q2[1])]
        )
        return direction / np.linalg.norm(direction)

    def diff1D(self, x, y):
        """
        Helper function used by getDirection
        """
        diff = y-x
        diff_pos = y+2*np.pi-x
        diff_neg = x+2*np.pi-y

        if abs(diff) <= abs(diff_pos) and abs(diff) <= abs(diff_neg):
            return diff
        elif abs(diff_pos) <= abs(diff_neg):
            return diff_pos
        else:
            return -diff_neg

    def planning(self, plot=False):
        """
        Rapidly exploring random tree (RRT)
        The method tries newly sampled direction until:
            1. The new node is close to the target in C space
            2. The number of nodes exceed maximum
        """
        iter = 0
        while iter <= self.max_iter:
            iter += 1

            if (random.random() <= self.goal_rate):
                # New direction points to goal
                q_new = self.goal
            else:
                # Sample uniformly between [0, 2pi]
                q_new = [random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi)]
            
            closest = self.getClosestNode(q_new)
            q_old = closest.q
            delta = self.getDirection(q_old, q_new)*self.step
            if (np.isnan(delta).any()):
                print ("Nan encountered. Start the next iteration.")
                continue
            q_new = self.roundOff([delta[0] + q_old[0], delta[1] + q_old[1]])
            
            newNode = Node(
                parent = closest,
                q = q_new
            )

            # Check collision
            if self.robot.collide(q_new, self.obstacles):
                print ("Collision at q =", q_new, ". Point discarded.")
                if plot:
                    self.plotInfeasiblePoint(newNode)
                continue
            else:
                print ("Safe at q =", q_new, ". Point accepted.")
                if plot:
                    self.plotFeasiblePoint(newNode)
                self.nodes.append(newNode)

            # Check goal
            d = self.distance(q_new, self.goal)
            print ("Distance to goal", d)
            if d < self.step:
                # Move to the goal and return the path
                print ("Goal reached!")
                goalNode = Node(
                    parent = newNode,
                    q = self.goal
                )
                self.nodes.append(goalNode)
                if plot:
                    self.plotFeasiblePoint(goalNode)
                pyplot.savefig("C_Space.png")
                return self.generatePath()

        raise RuntimeError("Unable to find a path. Please increase the number of iteration.")
    
    def plotInfeasiblePoint(self, node):
        """
        Plot a red line between q_old and q_new
        """
        pyplot.plot(node.q[0], node.q[1], 'rx')
        pyplot.title("Configuration Space of Double Pendulum")
        pyplot.xlim(0, 2*np.pi)
        pyplot.ylim(0, 2*np.pi)
        pyplot.pause(0.001)

    def plotFeasiblePoint(self, node):
        """
        Plot a green line between q_old and q_new
        """
        pyplot.plot(node.q[0], node.q[1], 'go')
        pyplot.title("Configuration Space of Double Pendulum")
        pyplot.xlim(0, 2*np.pi)
        pyplot.ylim(0, 2*np.pi)
        pyplot.pause(0.001)
    
    def generatePath(self):
        """
        Generate the path from self.nodes
        """
        node = self.nodes[-1]
        disp = []

        while node and node.parent:
            q0 = node.parent.q
            q1 = node.q
            disp.append(
                [self.diff1D(q0[0], q1[0]),
                 self.diff1D(q0[1], q1[1])]
            )
            node = node.parent
        disp.reverse()

        # Compute q without identification
        # This should be the input for MotionSolve
        path = [self.robot.q0]
        for d_i in disp:
            q0 = path[-1]
            q1 = [q0[0]+d_i[0], q0[1]+d_i[1]]
            path.append(q1)

        time = np.linspace(0, 10, len(path))
        with open('double_link.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(path)):
                writer.writerow([time[i], path[i][0], path[i][1]])

        return path

    def roundOff(self, q):
        """
        Make sure q_i in [0, 2pi]
        """
        if q[0] > 2*np.pi:
            q[0] -= 2*np.pi
        elif q[0] < 0:
            q[0] += 2*np.pi
        if q[1] > 2*np.pi:
            q[1] -= 2*np.pi
        elif q[1] < 0:
            q[1] += 2*np.pi
        return q

if __name__ == "__main__":
    # Create robot
    rect1 = Rectangle(1, 0.05)
    rect2 = Rectangle(1, 0.05)
    robot = DoublePendulum([0,0], [np.pi/2, 0], [rect1, rect2])

    # Create obstacles
    circle1 = Circle(r=0.5, pos=[1,1])
    circle2 = Circle(r=0.5, pos=[-1, 1.6])

    # Run Motion Planner
    planner = Planner(
        robot = robot,
        obstacles = [circle1, circle2],
        # obstacles = [],
        step_size = 0.1,
        goal_rate = 0.1,
        max_iter = 20000,
    )
    path = planner.planning(plot=True)
    print (path)
