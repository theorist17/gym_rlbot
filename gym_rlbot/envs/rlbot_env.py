import os
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as pb
import pybullet_data
import threading
from gym.spaces import Box, Discrete

joint_info_name = ["index", "name", "type", "qIndex", "uIndex", "flags",
                   "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                   "linkName", "axis", "parentFramePos", "parentFrameOrn", "parentIndex"]
joint_state_name = ["position", "velocity", "reactionForces", "torque"]


class DummyNet():
    def predict(self, state=[0] * 5):
        self.action = np.array([-0.8,-1.3,0.5,0,-1])
        return self.action


class RlbotEnv(gym.Env):
    metadata = {'render.modes': ['GUI', 'TUI']}

    def __init__(self, mode='GUI'):
        # gym
        self.observation_space = Box(low=np.array([-1.5]*5), high=np.array([1.5]*5), dtype=np.float16)
        self.action_space = Discrete(10)
        self.done = False
        self.goal = np.array([-0.8,-1.3,0.5,0,-1]) #

        # pybullet
        pb.connect(pb.GUI)  # if mode == 'GUI' else pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.81)  # everything should fall down
        pb.setTimeStep(0.000001) # default 240
        #pb.setRealTimeSimulation(1)  # we want to be faster than real time :)

        # plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pb.loadURDF("plane.urdf")

        # arm
        dir_name = os.path.dirname(__file__)
        self.arm_urdf_path = os.path.join(dir_name, '../../assets/Assem1/urdf/Assem1.urdf')
        self.arm = pb.loadURDF(self.arm_urdf_path, basePosition=[0, 0, 0],
                               baseOrientation=[-0.7071068, 0, 0, 0.7071068],
                               useFixedBase=1)
        self.num_joint = pb.getNumJoints(self.arm)

        #  button
        self.button_urdf_path = os.path.join(dir_name, '../../assets/button/urdf/simple_button.urdf')
        self.button = pb.loadURDF(self.button_urdf_path, basePosition=[0.1, 0, 0], useFixedBase=1)

    def step(self, action):
        done = False
        info = 0


        joint_states = pb.getJointStates(self.arm, jointIndices=range(self.num_joint))
        states = [joint[0] for joint in joint_states]
        prev_state = [round(state, 2) for state in states]
        pb.setJointMotorControlArray(
            bodyUniqueId=self.arm, jointIndices=range(self.num_joint), controlMode=pb.POSITION_CONTROL,
            targetPositions=action)
        for _ in range(240):
            pb.stepSimulation()

        joint_states = pb.getJointStates(self.arm, jointIndices=range(self.num_joint))
        states = [joint[0] for joint in joint_states]
        observation = [round(state, 2) for state in states]

        #print(observation, np.sum(np.abs(self.goal[0] - observation[0])))
        if np.sum(np.abs(self.goal- observation)) < np.sum(np.abs(self.goal - prev_state)):
            reward = 1 # 좋아
            #print('good', observation)
        else:
            reward = -2 #방향 나
            #print('bad', observation)
        if np.sum(np.abs(self.goal - observation)) < 0.5: #\성
            reward = 10
            print('success', observation)
            #print(observation, np.sum(np.abs(self.goal- observation)))
            done = True
        if np.sum(np.abs(self.goal - observation)) > 5: #실
            reward = -10
            print('fail', observation)
            #print(observation, np.sum(np.abs(self.goal - observation)))
            done = True

        return observation, reward, done, info

    def reset(self):
        pb.resetSimulation()

        pb.loadURDF("plane.urdf")
        pb.loadURDF(self.arm_urdf_path, basePosition=[0, 0, 0], baseOrientation=[-0.7071068, 0, 0, 0.7071068],
                    useFixedBase=1)
        pb.loadURDF(self.button_urdf_path, basePosition=[-0.07, -0.07, 0], useFixedBase=1)

        # pb.setJointMotorControlArray(
        #     bodyUniqueId=self.arm, jointIndices=range(self.num_joint), controlMode=pb.POSITION_CONTROL,
        #     targetPositions=np.array([0]*5))
        #
        joint_states = pb.getJointStates(self.arm, jointIndices=range(self.num_joint))
        states = [joint[0] for joint in joint_states]
        state = [round(state, 2) for state in states]
        state = np.array(state)

        return state
    def render(self):
        pass

    def close(self):
        pb.disconnect()


if __name__ == '__main__':
    env = RlbotEnv()
    model = DummyNet()

    state = env.reset()
    for _ in range(1000):
        score = 0
        action = model.predict(state)
        nextState, reward, done, info = env.step(action)

        state = nextState
        score += reward

        if done:
            print(score)
            env.reset()
        env.render()
    env.close()
