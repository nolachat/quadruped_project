# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
#
#log_dir = interm_dir + 'Perfect_FWD_1'
#log_dir = interm_dir + 'rapport_speedfixed_1'
#log_dir = interm_dir + 'rapport_speedfixedFlag_1'
#log_dir = interm_dir + 'rapport_speed07_2'
log_dir = interm_dir + 'flagrun_dp_space'



# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

env_config = {"motor_control_mode":"CARTESIAN_PD",#
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}


# env_config = {"motor_control_mode":"PD",
#                "task_env": "LR_COURSE_TASK",
#                "observation_space_mode": "DEFAULT"}

env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False
env_config['competition_env'] = False

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#
duration = 2 #[s]
TIME_STEP = 0.001
NSTEPS = int(duration//TIME_STEP)
NSIMS = 1
t = range(NSTEPS)

base_pos = np.zeros((2,len(t)))

speed = np.zeros((len(t)))
a_speed = np.zeros((len(t)))
distance = np.zeros((len(t)))
angle = np.zeros((len(t)))

FR = np.zeros((len(t),3))
FL = np.zeros((len(t),3))
RR = np.zeros((len(t),3))
RL = np.zeros((len(t),3))

energy = np.zeros((NSIMS,))
mass_offset = np.empty((NSIMS,4))

for j in range(NSIMS):
    for i in range(NSTEPS):
        action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards
        if dones:
            print('episode_reward', episode_reward)
            print('Final base position', info[0]['base_pos'])
            mass_offset[j] = env.envs[0].env._add_base_mass_offset()
            episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    # 
        base_lin_velocity = env.envs[0].env.robot.GetBaseLinearVelocity()
        speed[i] = np.linalg.norm(base_lin_velocity[:2])
        base_pos[:,i] = env.envs[0].env.robot.GetBasePosition()[0:2]

        angular_lin_velocity = env.envs[0].env.robot.GetBaseAngularVelocity()
        a_speed[i] = angular_lin_velocity[2]
        distance[i], angle[i] = env.envs[0].env.get_distance_and_angle_to_goal()

        _, FR_pos = env.envs[0].env.robot.ComputeJacobianAndPosition(0)
        FR[i,:] =  FR_pos
        _, FL_pos = env.envs[0].env.robot.ComputeJacobianAndPosition(1)
        FL[i,:] =  FL_pos
        _, RR_pos = env.envs[0].env.robot.ComputeJacobianAndPosition(2)
        RR[i,:] =  RR_pos
        _, RL_pos = env.envs[0].env.robot.ComputeJacobianAndPosition(3)
        RL[i,:] =  RL_pos

        for tau,vel in zip(env.envs[0].env._dt_motor_torques,env.envs[0].env._dt_motor_velocities):
            energy[j] += np.abs(np.dot(tau,vel)) * env.envs[0].env._time_step
    
# [TODO] make plots:

# Linear Speed plot
speed_mean = np.mean(speed)
speed_std = np.std(speed)

fig_speed, ax_speed = plt.subplots(figsize=(10, 4))
ax_speed.plot(t, speed, label='Speed $v$')
ax_speed.set_xlabel('Time')
ax_speed.set_ylabel('Speed')
plt.legend()
plt.title('Speed over Time')
ax_speed.text(0.05, 0.95, f'Mean: {speed_mean:.2f} m/s\nStd: {speed_std:.2f} m/s', 
        transform=ax_speed.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.show()

# Angular Speed and angle plot
fig_aspeed, ax_aspeed = plt.subplots(figsize=(10, 4))
ax_aspeed.plot(t, a_speed, label='Angular Speed $v$')
ax_aspeed.plot(t, angle, label='Angle $v$')
ax_aspeed.set_xlabel('Time')
plt.legend()
plt.title('Angular Speed over Time')
plt.show()

#Foot positions
plt.figure(figsize=(12, 18))
# Plot for X dimension
plt.subplot(3, 1, 1)
plt.plot(t, FR[:, 0], label='FR X Position', color='red')
plt.plot(t, FL[:, 0], label='FL X Position', color='blue')
plt.plot(t, RR[:, 0], label='RR X Position', color='green')
plt.plot(t, RL[:, 0], label='RL X Position', color='orange')
plt.xlabel('Time')
plt.ylabel('X Position')
plt.title('Legs X Position Over Time')
plt.legend()

# Plot for Y dimension
plt.subplot(3, 1, 2)
plt.plot(t, FR[:, 1], label='FR Y Position', color='red')
plt.plot(t, FL[:, 1], label='FL Y Position', color='blue')
plt.plot(t, RR[:, 1], label='RR Y Position', color='green')
plt.plot(t, RL[:, 1], label='RL Y Position', color='orange')
plt.xlabel('Time')
plt.ylabel('Y Position')
plt.title('Legs Y Position Over Time')
plt.legend()

# Plot for Z dimension
plt.subplot(3, 1, 3)
plt.plot(t, FR[:, 2], label='FR Z Position', color='red')
plt.plot(t, FL[:, 2], label='FL Z Position', color='blue')
plt.plot(t, RR[:, 2], label='RR Z Position', color='green')
plt.plot(t, RL[:, 2], label='RL Z Position', color='orange')
plt.xlabel('Time')
plt.ylabel('Z Position')
plt.title('Legs Z Position Over Time')
plt.legend()
plt.tight_layout()
plt.show()


# COT computation:
total_distance = np.zeros((NSIMS,))
cot = np.empty((NSIMS,))

g = 9.8 # in quadruped_gym_env.py
m = sum(env.envs[0].env.robot.GetTotalMassFromURDF()) # in quadruped.py

for i in range(NSIMS):

    # distance computation
    for step in range(NSIMS[i]-1):
        total_distance[i] += np.linalg.norm(base_pos[:,step+1,i]-base_pos[:,step,i])

    if speed_mean != 0:
        cot[i] = energy[i]/(m*g*speed_mean)

    if i == NSIMS:
        print("cot mean", np.mean(cot))
        print("cot std", np.std(cot))

print("==========CoT Calculations:============")
print("total_distance", total_distance)
print("total_energy", energy)
print("cot",cot)

# Robustness plots
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#sc = ax.scatter(mass_offset[:,0], mass_offset[:,1], mass_offset[:,2], s=5*mass_offset[:,3], c=last_step, cmap="tab20c_r", marker="s")

ax.set_title(f'Impact of the mass distribution on the episode length over {NSIMS} iterations')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
#cbar = fig.colorbar(sc, label="episode length")

plt.show()