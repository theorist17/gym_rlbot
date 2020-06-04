import os
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as pb
import pybullet_data
import threading
from gym.spaces import Box, Discrete

import threading
import logging
import time

joint_info_name = ["index", "name", "type", "qIndex", "uIndex", "flags",
                   "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                   "linkName", "axis", "parentFramePos", "parentFrameOrn", "parentIndex"]
joint_state_name = ["position", "velocity", "reactionForces", "torque"]


class DeviceManager():
    def __init__(self):
        self.client_id = None
        self.bot_id = None

        self.enc = None
        self.gyro = None
        self.tilt = None

        self.thread_read = threading.Thread(target=None)

        # TODO create camera, camera angle, and IR sensor
    def _read_encoder(self):
        '''
        get the signals from the motor encoders
        :return: sig_enc : numpy array of singals, can be set to None
        '''

        sig_enc = pb.getJointStates(self.bot_id, jointIndices=range(pb.getNumJoints(self.bot_id)))
        if sig_enc:
            self.enc = np.array([round(sig[0], 2) for sig in sig_enc])

    def _read_gyro(self):
        '''
        get the signals from the gyroscope sensor
        :return: sig_gyro
        '''
        gyro = (0, 0)
        self.gyro = gyro if gyro is not None else self.gyro

    def _read_tilt(self):
        '''
        get the signals from the tilt sensor
        :return: sig_tilt
        '''
        tilt = (0, 0)
        self.tilt = tilt if tilt is not None else self.tilt

    def _read(self):
        '''
        keeps reading and updating each value of encoder, gyro, tilt sensor whenever possible
        :return:
        '''

        while self.thread_read_on:
            self._read_encoder()
            self._read_gyro()
            self._read_tilt()

    def write_decoder(self, action):
        '''
        send the command signals to motor
        :param decode:
        :return: validity: whether the given action is in the acceptable range
        '''

        # TODO change action or a sequence of action into pwm signals
        signals = action

        pb.setJointMotorControlArray(
            bodyUniqueId=self.bot_id, jointIndices=range(pb.getNumJoints(self.bot_id)), controlMode=pb.POSITION_CONTROL,
            targetPositions=signals)

        return True

    def start_device_manager(self, client_id, bot_id):

        self.client_id = client_id
        self.bot_id = bot_id

        assert pb.isConnected(self.client_id) == True

        while self.enc is None:
            self._read_encoder()
        while self.gyro is None:
            self._read_gyro()
        while self.tilt is None:
            self._read_tilt()

        self.thread_read_on = True
        self.thread_read = threading.Thread(target=self._read)
        self.thread_read.start()

    def stop_device_manager(self):
        self.thread_read_on = False
        if self.thread_read.is_alive():
            self.thread_read.join()


mutex = threading.Lock()


class Localization():
    def __init__(self, device_manager, position=None, orientation=None, global_map=None):
        self.dm = device_manager
        self.pos = orientation
        self.ori = position
        self.gmap = global_map

        self.start_localization()

    def _odometry(self):
        '''
        calculate the current position/orientation of robot, from the encoders of motor and the values
        of both gyroscope sensor and tilt sensor
        '''

        global mutex
        while self.thread_odometry_on:
            enc = self.dm.enc
            gyro = self.dm.gyro
            tilt = self.dm.tilt

            # TODO perform odometry of calculating position, orientation of robot using the values of enc, gyro, tilt (오도메트리)

            mutex.acquire()
            self.pos = (0, 0, 0)
            self.ori = (0, 0, 0)
            mutex.release()

    def _perception(self):
        '''
        calculate the current position/orientation of robot as well as the local map of its surrounding,
        using camera, the tilt of camera, and infra-red sensor
        :return: position, orientation, local_map
        '''

        # TODO create position, orientation, global_map from camera, angle, and IR

        position = None
        orientation = None
        global_map = None

        return position, orientation, global_map

    def _match(self):
        '''
        perform odometry and perception to locate the robot and its surrounding
        :return: position, orientation, local_map
        '''

        global mutex
        while self.thread_localization_on:
            new_pos, new_ori, new_gmap = self._perception()

            # TODO match position and orientation from both odometry and perception, and update global_map using local_map

            mutex.acquire()
            self.pos = new_pos if new_pos else self.pos
            self.ori = new_ori if new_ori else self.ori
            self.gmap = new_gmap if new_gmap else self.gmap
            mutex.release()

    def start_localization(self):
        self.thread_odometry_on = True
        self.thread_odometry = threading.Thread(target=self._odometry)
        self.thread_odometry.start()

        self.thread_localization_on = True
        self.thread_localization = threading.Thread(target=self._match)
        self.thread_localization.start()

    def stop_localization(self):
        self.thread_odometry_on = False
        self.thread_odometry.join()

        self.thread_localization_on = False
        self.thread_localization.join()


# 0 BODY, LL1 (heap), LL2 (thigh), LL3 (jong), LL4 (ankle), LL5 (foot),
# LR1 (heap), LR2 (thigh), LR3 (jong), LR4 (ankle), LR5 (foot),
# AL1 (shoulder), AL2 (upper arm), AL3 (lower arm)
# AR1 (shoulder), AR2 (upper arm), AR3 (lower arm)

class RlbotEnv(gym.Env):
    '''
    mission : teach the agent for bipedal line tracing
    '''
    metadata = {'render.modes': ['GUI', 'TUI']}

    def __init__(self, mode='GUI'):
        # pybullet
        self.client_id = pb.connect(pb.GUI)  # if mode == 'GUI' else pb.connect(pb.DIRECT)

        # path
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        dir_name = os.path.dirname(__file__)
        self.button_urdf_path = os.path.join(dir_name, '../../assets/button/urdf/simple_button.urdf')
        self.bot_urdf_path = os.path.join(dir_name, '../../assets/rlbot_urdf/urdf/rlbot_urdf.xml')

        # camera setup
        pb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        # plane
        self.plane_id = pb.loadURDF("plane.urdf")

        # robot
        self.bot_base_pos = [0, 0, 0.05]
        self.bot_base_ori = pb.getQuaternionFromEuler([-1.5708, 0, 0])  # 1.5708 rad = 90 degree
        # TODO URDF 공식 문서 정독 http://wiki.ros.org/urdf/XML jointType -> revolute - a hinge joint that rotates along the axis and has a limited range specified by the upper and lower limits. soft_upper_limit 이런거 처리하기
        # TODO URDF Sensor의 Camera와 Ray 뭔지 연구하고 공유하기.

        #  button
        self.button_base_pos = [0.1, 0, 0]
        self.button_base_ori = pb.getQuaternionFromEuler([-1.5708, 0, 0])  # 1.5708 rad = 90 degree

        # gym
        # TODO update observation_space according to URDF
        self.observation_space = Box(low=np.array([-1.5] * 16), high=np.array([1.5] * 16), dtype=np.float16)
        self.action_space = Discrete(16)
        self.goal = np.array(0.0 * 16)

        # device manager
        self.dm = DeviceManager()

        # localization
        self.lc = Localization(self.dm)

    def step(self, action):
        if self.dm.write_decoder(action):
            pb.stepSimulation()

        observation = self.observe()
        reward = self.evaluate(observation)
        done = self.proceed(observation)
        info = None

        return observation, reward, done, info

    def reset(self):
        pb.resetSimulation()

        pb.setTimeStep(1./60)  # default 240 hz, Time steps larger than 1./60 can introduce instabilities for various reasons (deep penetrations, numerical integrator)
        pb.setGravity(0, 0, -10)
        pb.setRealTimeSimulation(0)

        # model load
        self.plane_id = pb.loadURDF("plane.urdf")
        self.bot_id = pb.loadURDF(self.bot_urdf_path, self.bot_base_pos, self.bot_base_ori)
        self.button_id = pb.loadURDF(self.button_urdf_path, self.button_base_pos, self.button_base_ori)

        # reset devices
        self.dm.stop_device_manager()
        self.lc.stop_localization()
        self.dm.start_device_manager(self.client_id, self.bot_id)
        self.lc.start_localization()

        next_state = self.observe()

        return next_state

    def render(self):
        pass

    def close(self):
        self.dm.thread_read_on = False
        self.lc.thread_odometry_on = False
        self.lc.thread_localization_on = False

        self.dm.thread_read.join()
        self.lc.thread_odometry.join()
        self.lc.thread_localization.join()

        pb.disconnect()

    def observe(self):
        '''
        observe sensors (motor-encoder/gyro/tilt) and localization (bot position/orientation, map) to create
        the next observation state to be passed to the agent

        :param odometry:
        :param localization:
        :return: observation
        '''

        enc = self.dm.enc
        tilt = self.dm.tilt
        gyro = self.dm.gyro
        pos = self.lc.pos
        ori = self.lc.ori
        gmap = self.lc.gmap

        state = np.array(enc)

        # TODO observation state definition

        observation = state

        return observation

    def evaluate(self, observation):
        '''
        evaluate observation to compute reward
        :return: reward
        '''

        reward = 0
        return reward

    def proceed(self, observation):
        '''
        determine whether to proceed with or abandon current episode
        return true if the agent
            0. succeeded to reach the goal, specified with XY position and XY orientation
            1. failed to find the right direction toward the goal
            2. failed to go straight once it find the right direction
            3. failed to balance while walking
            4. failed to get back up after falling down
        else false
        :return: done
        '''
        done = False
        return done

    def capture(self):
        # capture for getCameraImage
        pixelWidth = 1000
        pixelHeight = 1000
        camTargetPos = [0, 0, 0]
        camDistance = 0.5
        yaw = 0
        pitch = -10.0
        roll = 0
        upAxisIndex = 2
        viewMatrix = pb.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        pb.resetSimulation()


class DummyNet():
    def __init__(self):
        pass
    def predict(self, state):
        self.action = np.array([0] * len(state))
        return self.action


if __name__ == '__main__':
    env = RlbotEnv()
    model = DummyNet()

    while True:
        next_state = env.reset()
        score = 0

        for i in range(10000):
            action = model.predict(next_state)
            observation, reward, done, info = env.step(action)

            next_state = observation
            score += reward

            env.render()
            time.sleep(0.1)
            if done:
                print('score', score)
                break

    env.close()
