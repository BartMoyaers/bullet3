import numpy as np
import math
from pybullet_envs.deep_mimic.env.env import Env
from pybullet_envs.deep_mimic.env.action_space import ActionSpace
from pybullet_utils import bullet_client
import time
from pybullet_envs.deep_mimic.env import motion_capture_data
from pybullet_envs.deep_mimic.env import humanoid_stable_pd
from pybullet_envs.deep_mimic.env.goals import GoalType, Goal
from pybullet_envs.deep_mimic.env.humanoid_link_ids import HumanoidLinks
import pybullet_data
import pybullet as p1
import random


class PyBulletDeepMimicEnv(Env):

  def __init__(self, arg_parser=None, enable_draw=False, pybullet_client=None):
    super().__init__(arg_parser, enable_draw)
    self._num_agents = 1
    self._pybullet_client = pybullet_client
    self._isInitialized = False
    self._useStablePD = True
    self._arg_parser = arg_parser
    self.goal = self.getGoal()
    self.target_id = None

    self.reset()

    if self.goal.goal_type == GoalType.Strike:
      self.target_id = self.drawStrikeGoal()

  def reset(self):

    if not self._isInitialized:
      if self.enable_draw:
        self._pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
        #disable 'GUI' since it slows down a lot on Mac OSX and some other platforms
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)
      else:
        self._pybullet_client = bullet_client.BulletClient()

      self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
      z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
      self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                                     z2y,
                                                     useMaximalCoordinates=True)
      #print("planeId=",self._planeId)
      self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
      self._pybullet_client.setGravity(0, -9.8, 0)

      self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
      self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

      self._mocapData = motion_capture_data.MotionCaptureData()

      motion_file = self._arg_parser.parse_strings('motion_file')
      print("motion_file=", motion_file[0])

      motionPath = pybullet_data.getDataPath() + "/" + motion_file[0]
      #motionPath = pybullet_data.getDataPath()+"/motions/humanoid3d_backflip.txt"
      self._mocapData.Load(motionPath)
      timeStep = 1. / 240.
      useFixedBase = False
      self._humanoid = humanoid_stable_pd.HumanoidStablePD(self._pybullet_client, self._mocapData,
                                                           timeStep, useFixedBase, self._arg_parser)
      self._isInitialized = True

      self._pybullet_client.setTimeStep(timeStep)
      self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

      selfCheck = False
      if (selfCheck):
        curTime = 0
        while self._pybullet_client.isConnected():
          self._humanoid.setSimTime(curTime)
          state = self._humanoid.getState()
          #print("state=",state)
          pose = self._humanoid.computePose(self._humanoid._frameFraction)
          for i in range(10):
            curTime += timeStep
            #taus = self._humanoid.computePDForces(pose)
            #self._humanoid.applyPDForces(taus)
            #self._pybullet_client.stepSimulation()
          time.sleep(timeStep)
    #print("numframes = ", self._humanoid._mocap_data.NumFrames())
    #startTime = random.randint(0,self._humanoid._mocap_data.NumFrames()-2)
    rnrange = 1000
    rn = random.randint(0, rnrange)
    startTime = float(rn) / rnrange * self._humanoid.getCycleTime()
    self.t = startTime

    # Remove all the thrown objects
    self.removeThrownObjects()

    self._humanoid.setSimTime(startTime)

    self._humanoid.resetPose()
    #this clears the contact points. Todo: add API to explicitly clear all contact points?
    #self._pybullet_client.stepSimulation()
    self._humanoid.resetPose()
    # generate new goal
    self.goal.generateGoalData()
    self.needs_update_time = self.t - 1  #force update

  def get_num_agents(self):
    return self._num_agents

  def get_action_space(self, agent_id):
    return ActionSpace(ActionSpace.Continuous)

  def get_reward_min(self, agent_id):
    return 0

  def get_reward_max(self, agent_id):
    return 1

  def get_reward_fail(self, agent_id):
    return self.get_reward_min(agent_id)

  def get_reward_succ(self, agent_id):
    return self.get_reward_max(agent_id)

  #scene_name == "imitate" -> cDrawSceneImitate
  def get_state_size(self, agent_id):
    #cCtController::GetStateSize()
    #int state_size = cDeepMimicCharController::GetStateSize();
    #                     state_size += GetStatePoseSize();#106
    #                     state_size += GetStateVelSize(); #(3+3)*numBodyParts=90
    #state_size += GetStatePhaseSize();#1
    #197
    return 197

  def build_state_norm_groups(self, agent_id):
    #if (mEnablePhaseInput)
    #{
    #int phase_group = gNormGroupNone;
    #int phase_offset = GetStatePhaseOffset();
    #int phase_size = GetStatePhaseSize();
    #out_groups.segment(phase_offset, phase_size) = phase_group * Eigen::VectorXi::Ones(phase_size);
    groups = [0] * self.get_state_size(agent_id)
    groups[0] = -1
    return groups

  def build_state_offset(self, agent_id):
    out_offset = [0] * self.get_state_size(agent_id)
    phase_offset = -0.5
    out_offset[0] = phase_offset
    return np.array(out_offset)

  def build_state_scale(self, agent_id):
    out_scale = [1] * self.get_state_size(agent_id)
    phase_scale = 2
    out_scale[0] = phase_scale
    return np.array(out_scale)

  def get_goal_size(self, agent_id):
    return len(self.goal.getTFData())

  def get_action_size(self, agent_id):
    ctrl_size = 43  #numDof
    root_size = 7
    return ctrl_size - root_size

  def build_goal_norm_groups(self, agent_id):
    # Perform no normalization on goal data
    return np.array([-1] * len(self.goal.getTFData()))

  def build_goal_offset(self, agent_id):
    # no offset
    return np.array([0] * len(self.goal.getTFData()))

  def build_goal_scale(self, agent_id):
    # no scale
    return np.array([1] * len(self.goal.getTFData()))

  def build_action_offset(self, agent_id):
    out_offset = [0] * self.get_action_size(agent_id)
    out_offset = [
        0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, 0.0000000000,
        0.0000000000, -0.200000000, 0.0000000000, 0.0000000000, 0.00000000, -0.2000000, 1.57000000,
        0.00000000, 0.00000000, 0.00000000, -0.2000000, 0.00000000, 0.00000000, 0.00000000,
        -0.2000000, -1.5700000, 0.00000000, 0.00000000, 0.00000000, -0.2000000, 1.57000000,
        0.00000000, 0.00000000, 0.00000000, -0.2000000, 0.00000000, 0.00000000, 0.00000000,
        -0.2000000, -1.5700000
    ]
    #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
    #see cCtCtrlUtil::BuildOffsetScalePDSpherical
    return np.array(out_offset)

  def build_action_scale(self, agent_id):
    out_scale = [1] * self.get_action_size(agent_id)
    #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
    #see cCtCtrlUtil::BuildOffsetScalePDSpherical
    out_scale = [
        0.20833333333333, 1.00000000000000, 1.00000000000000, 1.00000000000000, 0.25000000000000,
        1.00000000000000, 1.00000000000000, 1.00000000000000, 0.12077294685990, 1.00000000000000,
        1.000000000000, 1.000000000000, 0.159235668789, 0.159235668789, 1.000000000000,
        1.000000000000, 1.000000000000, 0.079617834394, 1.000000000000, 1.000000000000,
        1.000000000000, 0.159235668789, 0.120772946859, 1.000000000000, 1.000000000000,
        1.000000000000, 0.159235668789, 0.159235668789, 1.000000000000, 1.000000000000,
        1.000000000000, 0.107758620689, 1.000000000000, 1.000000000000, 1.000000000000,
        0.159235668789
    ]
    return np.array(out_scale)

  def build_action_bound_min(self, agent_id):
    #see cCtCtrlUtil::BuildBoundsPDSpherical
    out_scale = [-1] * self.get_action_size(agent_id)
    out_scale = [
        -4.79999999999, -1.00000000000, -1.00000000000, -1.00000000000, -4.00000000000,
        -1.00000000000, -1.00000000000, -1.00000000000, -7.77999999999, -1.00000000000,
        -1.000000000, -1.000000000, -7.850000000, -6.280000000, -1.000000000, -1.000000000,
        -1.000000000, -12.56000000, -1.000000000, -1.000000000, -1.000000000, -4.710000000,
        -7.779999999, -1.000000000, -1.000000000, -1.000000000, -7.850000000, -6.280000000,
        -1.000000000, -1.000000000, -1.000000000, -8.460000000, -1.000000000, -1.000000000,
        -1.000000000, -4.710000000
    ]

    return out_scale

  def build_action_bound_max(self, agent_id):
    out_scale = [1] * self.get_action_size(agent_id)
    out_scale = [
        4.799999999, 1.000000000, 1.000000000, 1.000000000, 4.000000000, 1.000000000, 1.000000000,
        1.000000000, 8.779999999, 1.000000000, 1.0000000, 1.0000000, 4.7100000, 6.2800000,
        1.0000000, 1.0000000, 1.0000000, 12.560000, 1.0000000, 1.0000000, 1.0000000, 7.8500000,
        8.7799999, 1.0000000, 1.0000000, 1.0000000, 4.7100000, 6.2800000, 1.0000000, 1.0000000,
        1.0000000, 10.100000, 1.0000000, 1.0000000, 1.0000000, 7.8500000
    ]
    return out_scale

  def set_mode(self, mode):
    self._mode = mode

  def need_new_action(self, agent_id):
    if self.t >= self.needs_update_time:
      self.needs_update_time = self.t + 1. / 30.
      return True
    return False

  def record_state(self, agent_id):
    state = self._humanoid.getState()

    return np.array(state)

  def record_goal(self, agent_id):
    return np.array(self.goal.getTFData())

  def calc_reward(self, agent_id):
    kinPose = self._humanoid.computePose(self._humanoid._frameFraction)
    reward = self._humanoid.getReward(kinPose)

    mimic_weight = 0.7
    goal_weight = 0.3

    if self.goal.goal_type == GoalType.Strike:
      linkPos = self._humanoid.getLinkPosition(HumanoidLinks.rightAnkle)
      reward = mimic_weight * reward + goal_weight * self.calcStrikeGoalReward(linkPos)

    return reward

  def set_action(self, agent_id, action):
    #print("action=",)
    #for a in action:
    #  print(a)
    #np.savetxt("pb_action.csv", action, delimiter=",")
    self.desiredPose = self._humanoid.convertActionToPose(action)
    #we need the target root positon and orientation to be zero, to be compatible with deep mimic
    self.desiredPose[0] = 0
    self.desiredPose[1] = 0
    self.desiredPose[2] = 0
    self.desiredPose[3] = 0
    self.desiredPose[4] = 0
    self.desiredPose[5] = 0
    self.desiredPose[6] = 0
    target_pose = np.array(self.desiredPose)

    #np.savetxt("pb_target_pose.csv", target_pose, delimiter=",")

    #print("set_action: desiredPose=", self.desiredPose)

  def log_val(self, agent_id, val):
    pass

  def update(self, timeStep):
    #print("pybullet_deep_mimic_env:update timeStep=",timeStep," t=",self.t)
    self._pybullet_client.setTimeStep(timeStep)
    self._humanoid._timeStep = timeStep
    self.updateGoal(self._humanoid.getLinkPosition(HumanoidLinks.rightAnkle))
    if self.target_id is not None:
      self.updateDrawStrikeGoal()

    for i in range(1):
      self.t += timeStep
      self._humanoid.setSimTime(self.t)

      if self.desiredPose:
        kinPose = self._humanoid.computePose(self._humanoid._frameFraction)
        self._humanoid.initializePose(self._humanoid._poseInterpolator,
                                      self._humanoid._kin_model,
                                      initBase=True)
        #pos,orn=self._pybullet_client.getBasePositionAndOrientation(self._humanoid._sim_model)
        #self._pybullet_client.resetBasePositionAndOrientation(self._humanoid._kin_model, [pos[0]+3,pos[1],pos[2]],orn)
        #print("desiredPositions=",self.desiredPose)
        maxForces = [
            0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90,
            90, 90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100,
            100, 100, 60
        ]

        if self._useStablePD:
          usePythonStablePD = False
          if usePythonStablePD:
            taus = self._humanoid.computePDForces(self.desiredPose,
                                                desiredVelocities=None,
                                                maxForces=maxForces)
            #taus = [0]*43
            self._humanoid.applyPDForces(taus)
          else:
            self._humanoid.computeAndApplyPDForces(self.desiredPose,
                                                maxForces=maxForces)
        else:
          self._humanoid.setJointMotors(self.desiredPose, maxForces=maxForces)

        self._pybullet_client.stepSimulation()

  def set_sample_count(self, count):
    return

  def check_terminate(self, agent_id):
    return Env.Terminate(self.is_episode_end())

  def is_episode_end(self):
    isEnded = self._humanoid.terminates()
    #also check maximum time, 20 seconds (todo get from file)
    #print("self.t=",self.t)
    if (self.t > 20):
      isEnded = True
    return isEnded

  def check_valid_episode(self):
    #could check if limbs exceed velocity threshold
    return true

  def getKeyboardEvents(self):
    return self._pybullet_client.getKeyboardEvents()

  def isKeyTriggered(self, keys, key):
    o = ord(key)
    #print("ord=",o)
    if o in keys:
      return keys[ord(key)] & self._pybullet_client.KEY_WAS_TRIGGERED
    return False

  def hitWithObject(self):
    # Spawn an object with velocity hitting the model
    r = random.random()
    distance = 3.0
    rand_angle = r * 2 * math.pi - math.pi

    position, orientation = \
      self._humanoid.getSimModelBasePosition()

    # Remember, in bullet: Y direction is "up". X and Z direction are
    # in the horizontal plane.
    ball_position = [ distance * math.cos(rand_angle) + position[0],
                      position[1],
                      distance * math.sin(rand_angle)+ position[2]]

    visualShapeId = self._pybullet_client.createVisualShape(
      shapeType=self._pybullet_client.GEOM_SPHERE,
      radius=0.1)
    
    collisionShapeId = self._pybullet_client.createCollisionShape(
      shapeType=self._pybullet_client.GEOM_SPHERE,
      radius=0.1)

    body = self._pybullet_client.createMultiBody(
      baseMass=1,
      baseCollisionShapeIndex=collisionShapeId,
      baseVisualShapeIndex=visualShapeId,
      basePosition=ball_position)

    distance_scale = 10
    ball_velocity = [distance_scale * (position[i] - ball_position[i]) for i in range(3)]
    self._pybullet_client.resetBaseVelocity(body, linearVelocity=ball_velocity)

    self._humanoid.thrown_body_ids.append(body)

  def removeThrownObjects(self):
    # Disable gui so that object removal does not cause stutter
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING,0)

    for objectId in self._humanoid.thrown_body_ids:
      self._pybullet_client.removeBody(objectId)

    self._humanoid.thrown_body_ids = []

    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING,1)
  
  def getGoal(self):
    goal_type_str = self._arg_parser.parse_string("goal_type")
    return Goal(GoalType.from_str(goal_type_str))

  def calcStrikeGoalReward(self, linkPos):
    if self.goal.is_hit:
      return 1
    else:
      goalPos = self.goal.goal_data
      distanceSquared = sum([(x - y)**2 for (x, y) in zip(goalPos, linkPos)])
      return math.exp(-4*distanceSquared)

  def updateGoal(self, linkPos):
    if self.goal.goal_type == GoalType.Strike:
      goalPos = self.goal.goal_data
      distance = sum([(x - y)**2 for (x, y) in zip(goalPos, linkPos)]) ** 0.5

      if distance <= self.goal.hit_range:
        self.goal.is_hit = True

  def drawStrikeGoal(self):
    vis_id = self._pybullet_client.createVisualShape(
                shapeType=self._pybullet_client.GEOM_SPHERE,
                radius=0.2,
                rgbaColor=[1,0,0,0.5])

    obj_id = self._pybullet_client.createMultiBody(
                baseVisualShapeIndex=vis_id,
                basePosition=self.goal.goal_data)
    return obj_id

  def updateDrawStrikeGoal(self):
    current_pos = self._pybullet_client.getBasePositionAndOrientation(self.target_id)[0]
    target_pos = self.goal.goal_data

    if target_pos != current_pos:
      self._pybullet_client.resetBasePositionAndOrientation(
        self.target_id,
        target_pos,
        [0, 0, 0, 1]
      )
    if self.goal.is_hit != self.goal.is_hit_prev:
      self.goal.is_hit_prev = self.goal.is_hit
      if self.goal.is_hit:
        # Color green
        self._pybullet_client.changeVisualShape(
          self.target_id,
          -1,
          rgbaColor=[0, 1, 0, 0.5]
        )
      else:
        # Color red
        self._pybullet_client.changeVisualShape(
          self.target_id,
          -1,
          rgbaColor=[1, 0, 0, 0.5]
        )
