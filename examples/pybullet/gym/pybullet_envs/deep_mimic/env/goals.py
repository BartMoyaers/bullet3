from enum import Enum
from abc import ABC, abstractmethod
import random
import math
import pybullet as pb
import numpy

def randomVal(lowerBound, upperbound):
    difference = upperbound - lowerBound
    return lowerBound + random.random() * difference

class GoalType(Enum):
    NoGoal = 0
    Strike = 1
    TargetHeading = 2
    Throw = 3
    TerrainTraversal = 4

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
            
class StrikeGoal(Goal):
    def __init__(self):
        self.follow_rot = False
        self.is_hit_prev = False
        super().__init__(GoalType.Strike)
    
    def generateGoalData(self, modelPos=[0,0,0], modelOrient=[0,0,0,1]):
        # distance, height, rot
        distance = randomVal(0.6, 0.8)
        height = randomVal(0.8, 1.25)
        rot = randomVal(-1, 1) # radians
        
        self.is_hit = False
        
        # The max distance from the target counting as a hit
        self.hit_range = 0.2

        # Transform to xyz coordinates for placement in environment
        x = distance * math.cos(rot)
        y = distance * math.sin(rot)
        z = height

        # Y axis up, z axis in different direction
        self.goal_data = [-x, z, y]

        if self.follow_rot:
            # Take rotation of human model into account
            eulerAngles = pb.getEulerFromQuaternion(modelOrient)
            # Only Y angle matters
            eulerAngles = [0, eulerAngles[1], 0]
            yQuat = pb.getQuaternionFromEuler(eulerAngles)
            rotMatList = pb.getMatrixFromQuaternion(yQuat)
            rotMat = numpy.array([rotMatList[0:3], rotMatList[3:6], rotMatList[6:9]])
            vec = numpy.array(self.goal_data)
            rotatedVec = numpy.dot(rotMat, vec)
            self.world_pos = rotatedVec.tolist()

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
        return StrikeGoal()
    elif goal_type == GoalType.TargetHeading:
        return TargetHeadingGoal()
    else:
        raise NotImplementedError
