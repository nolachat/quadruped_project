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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838  # this is the hip length
# get correct hip sign (body right is negative)
sideSign = np.array([-1, 1, -1, 1])

env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging!
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      # record_video=True
                      )

# initialize Hopf Network, supply gait
# cpg = HopfNetwork(time_step=TIME_STEP)
gait = "TROT"
cpg = HopfNetwork(time_step=TIME_STEP, gait=gait)


TEST_STEPS = int(2 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [/TODO] initialize data structures to save CPG and robot states
amplitudes = np.zeros((4, len(t)))
phases = np.zeros((4, len(t)))
amplitudes_derivative = np.zeros((4, len(t)))
phases_derivative = np.zeros((4, len(t)))


# initialize arrays to store desired and current positions
desired_positions = np.zeros((4, 3, TEST_STEPS))
current_positions = np.zeros((4, 3, TEST_STEPS))
stored_desired_positions = np.zeros((3, TEST_STEPS))
stored_current_positions = np.zeros((3, TEST_STEPS))


# Sample Gains
# joint PD gains
kp = np.array([10, 100, 100])
kd = np.array([5, 5, 5])
# Cartesian PD gains
kpCartesian = np.diag([900]*3)  # 50
kdCartesian = np.diag([20]*3)  # 20


for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12)
    # get desired foot positions from CPG
    xs, zs = cpg.update()
    # [/TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()

    # loop through desired foot positions and calculate torques
    for i in range(4):
        # initialize torques for legi
        tau = np.zeros(3)
        # get desired foot i pos (xi, yi, zi) in leg frame
        leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
        # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)  # [/TODO]
        # Add joint PD contribution to tau for leg i (Equation 4)
        tau = tau + kp * (leg_q-q[3*i:3*i+3]) + kd * \
            (-dq[3*i:3*i+3])  # [/TODO]
        # add Cartesian PD contribution
        if ADD_CARTESIAN_PD:
            # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
            J, pos = env.robot.ComputeJacobianAndPosition(i)  # [/TODO]
            # Get current foot velocity in leg frame (Equation 2)
            v = J @ dq[3*i:3*i+3]  # [TODO]
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau = tau + J.T @ (np.matmul(kpCartesian, (leg_xyz-pos)) +
                               np.matmul(kdCartesian, (-v)))  # [/TODO]

        action[3*i:3*i+3] = tau
        if i == 0:
            # Assuming leg_q represents joint angles
            stored_desired_positions[:, j] = leg_q
            stored_current_positions[:, j] = q[3 * i:3 * i + 3]

            print(leg_q)

        # store desired and current positions
        desired_positions[i, :, j] = leg_xyz
        # print(desired_positions[i, :, j])
        current_positions[i, :, j] = pos

    env.step(action)

    # [/TODO] save any CPG or robot states

    amplitudes[:, j] = cpg.get_r()
    phases[:, j] = cpg.get_theta()
    amplitudes_derivative[:, j] = cpg.get_dr()
    phases_derivative[:, j] = cpg.get_dtheta()


#####################################################
# PLOTS
#####################################################


def legID_Name(legID):
    leg_names = {0: "FR Leg", 1: "FL Leg", 2: "RR Leg", 3: "RL Leg"}
    return leg_names.get(legID, "Leg ID not matching")


END_STEP = int(0.8 // (TIME_STEP))
START_STEP = int(0 // (TIME_STEP))


# Plot actual and desired positions for each joint
# tracking plot
leg_index = 0


# Create three subplots for the three joints
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot actual and desired positions for each joint
joint_labels = ['x', 'y', 'z']

for joint_index in range(3):
    axes[joint_index].plot(t[START_STEP:END_STEP], desired_positions[leg_index, joint_index, START_STEP:END_STEP],
                           label=f'Desired ', linestyle='--')
    axes[joint_index].plot(t[START_STEP:END_STEP], current_positions[leg_index, joint_index, START_STEP:END_STEP],
                           label=f'Actual', linestyle='-')
    axes[joint_index].set_ylabel(f'{joint_labels[joint_index]} (m)')

axes[2].set_xlabel('Time')
plt.legend()
plt.suptitle(
    f'Foot xyz positions for Front Right (FR) Leg', fontsize=16)
plt.show()


fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

joint_labels = [f'q{i}' for i in range(3)]

for joint_index in range(3):
    axes[joint_index].plot(t[START_STEP:END_STEP], stored_desired_positions[joint_index, START_STEP:END_STEP],
                           label=f'Desired {joint_labels[joint_index]}', linestyle='--')
    axes[joint_index].plot(t[START_STEP:END_STEP], stored_current_positions[joint_index, START_STEP:END_STEP],
                           label=f'Actual {joint_labels[joint_index]}', linestyle='-')
    axes[joint_index].set_ylabel(f'{joint_labels[joint_index]} (rad)')

axes[2].set_xlabel('Time')
plt.legend()
plt.suptitle(
    f'Joint position for Front Right (FR) Leg', fontsize=16)
plt.show()
# Create four subplots
# fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot each vector in a separate subplot
# for i in range(4):
#     axes[i].plot(t[START_STEP:END_STEP], amplitudes[i,
#                  START_STEP:END_STEP], label=f'Amplitude $r$')
#     axes[i].plot(t[START_STEP:END_STEP],
#                  phases[i, START_STEP:END_STEP], label=f'Phase $\\theta$ ')
#     axes[i].plot(t[START_STEP:END_STEP], amplitudes_derivative[i,
#                  START_STEP:END_STEP], label=f'Amplitude Derivative $\\dot{{r}}$')
#     axes[i].plot(t[START_STEP:END_STEP], phases_derivative[i,
#                  START_STEP:END_STEP], label=f'Phase Derivative $\\dot{{\\theta}}$')
#     axes[i].set_ylabel(f'{legID_Name(i)}')

# axes[3].set_xlabel('Time')
# plt.legend()
# plt.suptitle(
#     f'CPG states ($r, \\theta, \\dot{{r}}, \\dot{{\\theta}}$) for a {gait} gait', fontsize=16)
# plt.show()

""" 
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot positions on each subplot
for i in range(4):
    x = amplitudes[i, :] * np.cos(phases[i, :])
    y = amplitudes[i, :] * np.sin(phases[i, :])

    axes[i].plot(x, y)
    axes[i].set_title(legID_Name(i))
    axes[i].set_ylabel(r'$Y = r \cdot \sin({\theta})$')
    axes[i].set_xlabel(r'$X = r \cdot \cos({\theta})$')

plt.suptitle(f'Position Plots for Each Amplitude and Phase Pair with gait: {gait}', fontsize=16)
plt.tight_layout()
plt.show() """
# # To plot the desired and current positions over time:
# fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# for i in range(4):
#     axes[i].plot(t[START_STEP:END_STEP], desired_positions[i,
#                  0, START_STEP:END_STEP], label='Desired X')
#     axes[i].plot(t[START_STEP:END_STEP], current_positions[i,
#                  0, START_STEP:END_STEP], label='Current X')
#     axes[i].set_ylabel(f'{legID_Name(i)}')

# axes[3].set_xlabel('Time')
# plt.legend()
# plt.suptitle(f'Desired and Current X Positions for Each Leg', fontsize=16)
# plt.show()

# # Plot the stored joint positions after the loop
# for joint_index in range(3):
#     axes[joint_index].plot(np.arange(time_intervals), stored_positions["desired"][joint_index, :],
#                            marker='o', linestyle='-', label=f'Desired {joint_labels[joint_index]}')
#     axes[joint_index].plot(np.arange(time_intervals), stored_positions["current"][joint_index, :],
#                            marker='x', linestyle='-', label=f'Current {joint_labels[joint_index]}')
#     axes[joint_index].set_ylabel(f'{joint_labels[joint_index]} (degrees)')
