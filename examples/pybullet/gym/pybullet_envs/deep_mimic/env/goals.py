from enum import Enum
from abc import ABC, abstractmethod
import random
import math
import pybullet as pb
import numpy

class RandomBounds:
    def __init__(self, lower_bound: float, upper_bound: float):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.difference = upper_bound - lower_bound

    def generate(self) -> float:
        return self.lower_bound + random.random() * self.difference
    
def randomVal(lowerBound, upperbound):
    difference = upperbound - lowerBound
    return lowerBound + random.random() * difference

class GoalType(Enum):
    NoGoal = 0
    Strike = 1
    TargetHeading = 2
    Throw = 3
    TerrainTraversal = 4
    Grab = 5

    @staticmethod
    def from_str(goal_type_str):
        if goal_type_str == '':
            return GoalType.NoGoal
        else:
            try:
                return GoalType[goal_type_str]
            except:
                raise NotImplementedError

class Goal(ABC):
    def __init__(self, goal_type: GoalType):
        self.goal_type = goal_type
        self.generateGoalData()

    @abstractmethod
    def generateGoalData(self, modelPos=[0,0,0], modelOrient=[0,0,0,1]):
        pass

    @abstractmethod
    def getTFData(self):
        pass

class NoGoal(Goal):
    def __init__(self):
        super().__init__(GoalType.NoGoal)

    def generateGoalData(self, modelPos=[0,0,0], modelOrient=[0,0,0,1]):
        self.goal_data = []

    def getTFData(self):
        return self.goal_data

class Strike(Goal):
    def __init__(self,
                distance_RB: RandomBounds,
                height_RB: RandomBounds,
                rot_RB: RandomBounds,
                hit_range = 0.2,
                follow_rot = True):
        self.follow_rot = follow_rot
        self.is_hit_prev = False
        self.distance_RB = distance_RB
        self.height_RB = height_RB
        self.rot_RB = rot_RB
        self.hit_range = hit_range
        super().__init__(GoalType.Strike)
    
    def generateGoalData(self,
                        modelPos=[0,0,0],
                        modelOrient=[0,0,0,1]):
        # distance, height, rot
        distance = self.distance_RB.generate()
        height = self.height_RB.generate()
        rot = self.rot_RB.generate() # radians

        self.is_hit = False

        # Transform to xyz coordinates for placement in environment
        x = distance * math.cos(rot)
        y = distance * math.sin(rot)
        z = height

        # Y axis up, z axis in different direction
        self.goal_data = [-x, z, y]

        if self.follow_rot:
            # Take rotation of robot model into account
            rotMatList = pb.getMatrixFromQuaternion(modelOrient)
            rotMat = numpy.array([rotMatList[0:3], rotMatList[3:6], rotMatList[6:9]])
            vec = numpy.array(self.goal_data)
            rotatedVec = numpy.dot(rotMat, vec)
            self.world_pos = rotatedVec.tolist()

            # Correct distance and height after rotation
            curr_distance = (self.world_pos[0] ** 2 + self.world_pos[2] ** 2) ** 0.5
            factor = distance / curr_distance
            self.world_pos[0] *= factor
            self.world_pos[1] = height
            self.world_pos[2] *= factor

            # Add translation of agent in world
            self.world_pos =    [   self.world_pos[0] + modelPos[0],
                                    self.world_pos[1],
                                    self.world_pos[2] + modelPos[2]]
        else:
            self.world_pos = [-x + modelPos[0], z, y + modelPos[2]]

    def getTFData(self):
        x = 0.0
        if self.is_hit:
            x = 1.0
        return [x] + self.goal_data

class Kick(Strike):
    def __init__(self):
        distance_RB = RandomBounds(0.6, 0.8)
        height_RB = RandomBounds(0.8, 1.25)
        rot_RB = RandomBounds(-1, 1)
        super().__init__(distance_RB, height_RB, rot_RB, hit_range=0.2)

class Grab(Strike):
    def __init__(self):
        distance_RB = RandomBounds(0.6, 0.8)
        height_RB = RandomBounds(0.8, 1.1)
        rot_RB = RandomBounds(3.14159 - 0.5, 3.14159 + 0.5)
        hit_range = 0.1
        super().__init__(distance_RB, height_RB, rot_RB, hit_range=hit_range, follow_rot=False)

class TargetHeadingGoal(Goal):
    def __init__(self):
        super().__init__(GoalType.TargetHeading)

    def generateGoalData(self, modelPos=[0,0,0], modelOrient=[0,0,0,1]):
        # Direction: 2D unit vector
        # speed: max speed
        random_rot = random.random() * 2 * math.pi
        x = math.cos(random_rot)
        y = math.sin(random_rot)
        velocity = randomVal(0, 0.5)
        self.goal_data = [x, y, velocity]

    def getTFData(self):
        return self.goal_data

def createGoal(goal_type: GoalType) -> Goal:
    if goal_type == GoalType.NoGoal:
        return NoGoal()
    elif goal_type == GoalType.Strike:
        return Kick()
    elif goal_type == GoalType.TargetHeading:
        return TargetHeadingGoal()
    elif goal_type == GoalType.Grab:
        return Grab()
    else:
        raise NotImplementedError
