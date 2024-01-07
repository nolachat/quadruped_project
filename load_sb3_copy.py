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
log_dir = interm_dir + 'CPG_desireless_v2_Goal_4'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

env_config = {"motor_control_mode":"CPG",
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}


# env_config = {"motor_control_mode":"PD",
#                "task_env": "LR_COURSE_TASK",
#                "observation_space_mode": "DEFAULT"}

env_config['render'] = False
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

# [TODO] initialize arrays to save data from simulation

num_sim = 1 # number of times to run the sicmulations
MAX_STEPS = 1000 # maximum number of steps to plot, plot is shorted to number of steps done by first sim

t = range(MAX_STEPS)
last_step = np.empty((num_sim,), dtype=int)

amplitudes = np.zeros((4,len(t)))
phases = np.zeros((4,len(t)))
amplitudes_derivative = np.zeros((4,len(t)))
phases_derivative = np.zeros((4,len(t)))
stance_indication = np.zeros((4,len(t)))


base_speed = np.empty((len(t),num_sim))

base_pos = np.zeros((2,len(t),num_sim))

dy = np.zeros((2,len(t)))

w_z = np.zeros((len(t),))

goal_angle = np.zeros((len(t),))

energy = np.zeros((num_sim,))

mass_offset = np.empty((num_sim,4))

# For the simulation
i = 0 # timesteps counter
sim = 0 # sim counter
episode_reward = 0

while sim < num_sim :

    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    if dones:

        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])

        episode_reward = 0
        mass_offset[sim] = env.envs[0].env.mass_offset
        last_step[sim] = i

        sim = sim + 1
        i = 0
        continue

    # [TODO] save data from current robot states for plots
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition()

    if sim == 0:
        # CPG states
        amplitudes[:,i] = env.envs[0].env._cpg.get_r()
        phases[:,i] = env.envs[0].env._cpg.get_theta()
        amplitudes_derivative[:,i] = env.envs[0].env._cpg.get_dr()
        phases_derivative[:,i] = env.envs[0].env._cpg.get_dtheta()

        _, p0 = env.envs[0].env.robot.ComputeJacobianAndPosition(0)
        dy[0,i] =  p0[1]
        _, p1 = env.envs[0].env.robot.ComputeJacobianAndPosition(1)
        dy[1,i] =  p1[1]

        w_z[i] = env.envs[0].env.robot.GetBaseAngularVelocity()[2]

        d, goal_angle[i] = env.envs[0].env.get_distance_and_angle_to_goal()

        stance_indication[:,i] = env.envs[0].env.robot.GetContactInfo()[3]

    for tau,vel in zip(env.envs[0].env._dt_motor_torques,env.envs[0].env._dt_motor_velocities):
        energy[sim] += np.abs(np.dot(tau,vel)) * env.envs[0].env._time_step

    base_pos[:,i,sim] = env.envs[0].env.robot.GetBasePosition()[0:2]
    base_speed[i,sim] = np.linalg.norm(env.envs[0].env.robot.GetBaseLinearVelocity()[0:2])

    i = i + 1 

# [TODO] make plots:
legID_Name = {0: "FR Leg", 1: "FL Leg", 2: "RR Leg", 3: "RL Leg"}

START_STEP = 0
PlOT_STEPS = last_step[0]

# Create four subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot each vector in a separate subplot
for i in range(4):
    axes[i].plot(t[START_STEP:PlOT_STEPS], amplitudes[i, START_STEP:PlOT_STEPS], label=f'Amplitude $r$')
    axes[i].plot(t[START_STEP:PlOT_STEPS], phases[i, START_STEP:PlOT_STEPS], label=f'Phase $\\theta$ ')
    axes[i].plot(t[START_STEP:PlOT_STEPS], amplitudes_derivative[i, START_STEP:PlOT_STEPS], label=f'Amplitude Derivative $\\dot{{r}}$')
    axes[i].plot(t[START_STEP:PlOT_STEPS], phases_derivative[i, START_STEP:PlOT_STEPS], label=f'Phase Derivative $\\dot{{\\theta}}$')
    axes[i].set_ylabel(f'{legID_Name[i]}')

    for j in t[START_STEP:PlOT_STEPS-1]:
        # leave contact
        if stance_indication[i,j] and not stance_indication[i,j+1]:
            axes[i].axvline(j, color='teal')
        # enter contact
        if not stance_indication[i,j] and stance_indication[i,j+1]:
            axes[i].axvline(j, color='mediumpurple')

axes[3].set_xlabel('Time')
plt.legend()
plt.suptitle(f'CPG states ($r, \\theta, \\dot{{r}}, \\dot{{\\theta}}$)', fontsize=16)
plt.show()


## Plotting velocity state
mean_speed = [np.mean(base_speed[0:last_step[n],n]) for n in range(num_sim) ]
std = [np.std(base_speed[0:last_step[n],n]) for n in range(num_sim) ]

fig, ax = plt.subplots()

ax.plot(base_speed[START_STEP:PlOT_STEPS,0], label='Speed |v(x,y)|')
ax.set_title('Evolution of speed')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Speed [m/s]')
ax.legend()
ax.grid(True)

# Add mean and std as text
ax.text(0.05, 0.95, f'Mean: {mean_speed[0]:.2f} m/s\nStd: {std[0]:.2f} m/s', 
        transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.show()

# Creating a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

# Plotting dy components in the first subplot
ax1.plot(t[START_STEP:PlOT_STEPS], dy[0, START_STEP:PlOT_STEPS], label='lateral displacement on FR Leg [m]')
ax1.plot(t[START_STEP:PlOT_STEPS], dy[1, START_STEP:PlOT_STEPS], label='lateral displacement on FL Leg [m]')
ax1.legend()

# Plotting w_z and goal_angle in the second subplot
ax2.plot(w_z[START_STEP:PlOT_STEPS], label='rotation speed, w_z [rad/s]')
ax2.plot(goal_angle[START_STEP:PlOT_STEPS], label='Angular difference to the goal [rad]')
ax2.set_xlabel('Timesteps')
ax2.axhline(0)
ax2.legend()

# Adding a title for the entire figure
fig.suptitle('Plots of dy[FR], dy[FL], w_z, and goal_angle')

# Display the plot
plt.show()

# COT computation:
total_distance = np.zeros((num_sim,))
cot = np.empty((num_sim,))

g = 9.8 # in quadruped_gym_env.py
m = sum(env.envs[0].env.robot._total_mass_urdf) # in quadruped.py

for i in range(num_sim):

    # distance computation
    for step in range(last_step[i]-1):
        total_distance[i] += np.linalg.norm(base_pos[:,step+1,i]-base_pos[:,step,i])

    if mean_speed[i] != 0:
        cot[i] = energy[i]/(m*g*mean_speed[i])

print("==========CoT Calculations:============")
print("total_distance", total_distance)
print("total_energy", energy)
print("cot",cot)

if num_sim > 1:
    print("cot mean", np.mean(cot))
    print("cot std", np.std(cot))

# Robustness plots
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(mass_offset[:,0], mass_offset[:,1], mass_offset[:,2], s=5*mass_offset[:,3], c=last_step, cmap="cool", marker="s")

ax.set_title('Impact of the mass distribution on the episode length')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
cbar = fig.colorbar(sc, label="episode length")

plt.show()
