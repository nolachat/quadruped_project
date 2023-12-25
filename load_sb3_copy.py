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

import textwrap

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
# log_dir = interm_dir + '121523095438'
log_dir = interm_dir + 'Extended_CPG_v02_track_speed_command'
# log_dir = interm_dir + 'v=1'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

env_config = {"motor_control_mode":"CPG",
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}


# env_config = {"motor_control_mode":"PD",
#                "task_env": "LR_COURSE_TASK",
#                "observation_space_mode": "DEFAULT"}

env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
# env_config['competition_env'] = True

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
env.training = True    # do not update stats at test time
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
t = range(NSTEPS)

amplitudes = np.zeros((4,len(t)))
phases = np.zeros((4,len(t)))
amplitudes_derivative = np.zeros((4,len(t)))
phases_derivative = np.zeros((4,len(t)))

base_speed = np.zeros((1,len(t)))

for i in range(NSTEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    # 
    amplitudes[:,i] = env.envs[0].env._cpg.get_r()
    phases[:,i] = env.envs[0].env._cpg.get_theta()
    amplitudes_derivative[:,i] = env.envs[0].env._cpg.get_dr()
    phases_derivative[:,i] = env.envs[0].env._cpg.get_dtheta()
    base_speed[:,i] = env.envs[0].env.robot.GetBaseLinearVelocity()[0]
    print("desired_velocity:", str(env.envs[0].env.desired_velocity))

# [TODO] make plots:

# PlOT_STEPS = int(1.2 // (TIME_STEP))
# START_STEP = int(0 // (TIME_STEP))
    
PlOT_STEPS = 600
START_STEP = 400


legID_Name = {0: "FR Leg", 1: "FL Leg", 2: "RR Leg", 3: "RL Leg"}

# Create four subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot each vector in a separate subplot
for i in range(4):
    axes[i].plot(t[START_STEP:PlOT_STEPS], amplitudes[i, START_STEP:PlOT_STEPS], label=f'Amplitude $r$')
    axes[i].plot(t[START_STEP:PlOT_STEPS], phases[i, START_STEP:PlOT_STEPS], label=f'Phase $\\theta$ ')
    axes[i].plot(t[START_STEP:PlOT_STEPS], amplitudes_derivative[i, START_STEP:PlOT_STEPS], label=f'Amplitude Derivative $\\dot{{r}}$')
    axes[i].plot(t[START_STEP:PlOT_STEPS], phases_derivative[i, START_STEP:PlOT_STEPS], label=f'Phase Derivative $\\dot{{\\theta}}$')
    axes[i].set_ylabel(f'{legID_Name[i]}')

axes[3].set_xlabel('Time')
plt.legend()
plt.suptitle(f'CPG states ($r, \\theta, \\dot{{r}}, \\dot{{\\theta}}$)', fontsize=16)
plt.show()

plt.plot()

mean_speed = np.mean(base_speed[0])
std = np.std(base_speed[0])

# Plotting
plt.plot(t, base_speed[0], label='Speed along x')
plt.title('Speed along x axis')
plt.xlabel('Timesteps')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid(True)

text = fr'Results for speed tracking' + '\n' + fr'$\mu$ = {mean_speed:.2f}, $\sigma$ = {std:.2f}'

plt.text(0.5, 0.1, text, transform=plt.gca().transAxes, ha='center', va='bottom', fontsize=12, wrap=True)

plt.show()