from enum import Enum
import random
import math

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

class Goal:
    def __init__(self, goal_type: GoalType):
        self.goal_type = goal_type
        self.generateGoalData()
        self.is_hit_prev = False

    def generateGoalData(self, modelPos=[0,0]):
        if self.goal_type == GoalType.NoGoal:
            self.goal_data = []

        elif self.goal_type == GoalType.Strike:
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
            self.world_pos = [-x + modelPos[0], z, y + modelPos[1]]

        elif self.goal_type == GoalType.TargetHeading:
            # Direction: 2D unit vector
            # speed: max speed
            random_rot = random.random() * 2 * math.pi
            x = math.cos(random_rot)
            y = math.sin(random_rot)
            velocity = randomVal(0, 0.5)
            self.goal_data = [x, y, velocity]

        elif self.goal_type == GoalType.Throw:
            # TODO
            raise NotImplementedError
        elif self.goal_type == GoalType.TerrainTraversal:
            # TODO
            raise NotImplementedError

    def getTFData(self):
        if self.goal_type == GoalType.Strike:
            x = 0.0
            if self.is_hit:
                x = 1.0
            return [x] + self.goal_data
            