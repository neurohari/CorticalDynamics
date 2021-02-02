# -*- coding: utf-8 -*-
"""
Created on February 2021

@author: hariteja1992@gmail.com

This code optimizes the neural networks for posture perturbation task.

Dependecies: 
    1. biomechanical arm model from "MuscularArm.py"
    2. Network class from "FeedbackController.py"
    3. Scientific libraries such as numpy, matplotlib etc.,
    4. PyTorch deep-learning library from: https://pytorch.org/get-started/locally/

code structure:
    1. define the time duration, feedback properties, task goal 
    2. initialize the neural network and optimizer
    3. While optimizing:
        a. run the control and movement for one time-instance
        b. collect the trajectory and network information
        c. backpropagate the error to compute the gradients (dJ/dw)
        d. change weights using ADAM rule
        
variables of interest for plotting are stored in "collector_" variables inside the 
"feedback_controller" class
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from NetworkClass import feedback_controller, costcriterionPosture

use_cuda = 'false'
device = torch.device('cuda:0' if use_cuda else 'cpu')
device = 'cpu'

# Time-settings
dh = 0.01 # 10 ms sim-step
T = 3.0 # 3 seconds simulation duration (300 time-steps)
#time_seq = np.linspace(0.0,T/dh-1,T/dh)
# if the time_seq def above yields a float-int error use the following line instead
time_seq = np.linspace(0.0,299,300)
perturbation_time = 100 # around t = 1sec apply perturbation

# Task scenario settings
num_joints = 2
num_torque_combinations = 9


# spatial target in radians (Home location for the hand/end-effector)
xtarg = np.array([0.4368]) # shoulder angle
ytarg = np.array([1.4464]) # elbow angle


# Define the tensors to carry task information. 
# each tensor is [T, C, N].Where
# T = time sequence length
# C = number of task conditions (in this case number of different perturbations)  
# N = signal dimensionality (number of neurons (or) number of sensory-feedback components)
# input_targ_seq as network spatial target, perturb_seq as the external torques n [sho, elb] joint-directions
input_targ_seq = np.zeros([time_seq.size, num_torque_combinations, num_joints])
perturb_seq = np.zeros([time_seq.size, num_torque_combinations, num_joints])
test_perturb_seq = np.zeros([time_seq.size, num_torque_combinations, num_joints])

# apply perturbations (radial in shoulder-elbow space)
pert_angles = np.array(np.linspace(0, 360, num_torque_combinations))
test_pert_angles = np.random.uniform(low=0.0, high=360.0, size=(num_torque_combinations,))
# perturb with 0.2 Nm of l2-norm joint-torque
xpert = 0.2 * np.cos(np.deg2rad(pert_angles))
ypert = 0.2 * np.sin(np.deg2rad(pert_angles))
xpert[-1] = 0.0 # include a no perturbation condition at the end
ypert[-1] = 0.0 # include a no perturbation condition at the end


## write the temporal and conditional aspects of the task into 3-D (T by C by N) tensor format
input_targ_seq[:, :, 0] = xtarg[:] # target shoulder joint config
input_targ_seq[:, :, 1] = ytarg[:] # target elbow joint config
perturb_seq[perturbation_time:, :num_torque_combinations, 0] = xpert # shoulder perturbation info
perturb_seq[perturbation_time:, :num_torque_combinations, 1] = ypert # elbow perturbation info

# wrap the numpy input sequence into tensors for pytorch
input_targ_seq = torch.from_numpy(input_targ_seq) 
perturb_seq = torch.from_numpy(perturb_seq)
test_perturb_seq = torch.from_numpy(test_perturb_seq)
input_targ_seq = input_targ_seq.float().to(device)
perturb_seq = perturb_seq.float().to(device)

# specify home location as a torch tensor
home_jstate = input_targ_seq[0, :, :]

# Which kind of network? RNN or FNN
rec_connection_status = 'True' # enter 'False' to switch off recurrent connections

# simulation and file settings for training/optimizing the networks
max_simulations = 1
EPOCHS = 2000
num_optimizations = 1
file_name = 'Posture_'
fb_type = 'posvelmus_fb_'

# The while loop is for training multiple networks
# The for-loop inside is to train one network for fixed EPOCHS or until a reasonabe cost is obtained.
while num_optimizations <= max_simulations:
    # initilize a neural network instance and send it to device = 'cpu'
    posture_sim = feedback_controller(dh, home_jstate)
    posture_sim.num_inplayer_neurons =500
    posture_sim.num_outplayer_neurons = 500
    posture_sim.num_muscles = 6
    posture_sim.feedback_delay = 5 # 50ms delay (or 5 time-steps)
    posture_sim.num_feedbacksignals = 12 # 2 target pos + 6 muscle_fb + 2 actual pos + 2 actual vel
    posture_sim.perturbation_time = perturbation_time
    posture_sim.num_torque_combinations = num_torque_combinations
    posture_sim.float()
    posture_sim.to(device)
    
    # decide if the network has recurrent connections or not
    posture_sim.rec_connection_status = 'False'
    
    # Set the neural newtork paramters to be optimized.  
    lrate = 3.0e-4
    if posture_sim.rec_connection_status == 'True':
        optim_params = list(posture_sim.inplayer.parameters()) + list(posture_sim.outplayer.parameters()) + list(posture_sim.musclelayer.parameters())
        optim_params += list(posture_sim.inplayerself.parameters()) + list(posture_sim.outplayerself.parameters())
        print('REC network selected with optimizable recurrent connections')
    else:
        optim_params = list(posture_sim.inplayer.parameters()) + list(posture_sim.outplayer.parameters()) + list(posture_sim.musclelayer.parameters())
        print('NOREC network selected with no recurrent connections')
    # define the optimizer
    optimizer = optim.Adam(optim_params, lr=lrate, weight_decay=1e-8) # good for the RNN case lr = 1.0e-4

    # Loss value holders
    cost_curve_for_plotting = np.empty(0)
    total_loss_for_plotting = np.zeros(EPOCHS)
    
    # Run the optimization until the end of epochs (or can be modified to run 
    # until a satisfactory error is reached)
    for epoch in range(1, EPOCHS):
        # reset the arm movement simulation to initial states of network and the arm
        posture_sim.resetsim()
        
        # reset the optimizer gradients
        optimizer.zero_grad()
        
        # Run the arm control simulation for the duration set as time_seq and
        # collect the joint, muscle and network information
        joint_kinematics = posture_sim.forward(input_targ_seq, perturb_seq)
        x_pos = joint_kinematics[:, :, 0] # shoulder displacement
        y_pos = joint_kinematics[:, :, 1] # elbow displacement
        x_vel = joint_kinematics[: ,:, 2] # shoulder velocity
        y_vel = joint_kinematics[:, :, 3] # elbow velocity
        
        # Compute the loss function defined by "costcriterionPosture"
        loss_x = costcriterionPosture(posture_sim, input_targ_seq[:, :, 0], x_pos, x_vel)
        loss_y = costcriterionPosture(posture_sim, input_targ_seq[:, :, 1], y_pos, y_vel)
        loss = loss_x + loss_y
        
        # Backpropagate the loss to compute loss gradients w.r.t network paramters
        loss.backward() # does backprop and calculates gradients
        cost_curve_for_plotting = np.append(cost_curve_for_plotting, loss.item())
        
        # Update the newtork weights
        optimizer.step()
        
        
        total_loss_for_plotting[epoch] = loss.item()
        
        
        if epoch%100 == 0:
            print("Cur instance Loss: {:.6f}".format(loss.item()), end=' ')
            print(' after Epoch number: {}/{}...'.format(epoch, EPOCHS), end=' ')
            print('num_optimizations: {}/{}'.format(num_optimizations, max_simulations, ))
            
        if epoch % 100 == 0:
            plt.figure()
            plt.plot(posture_sim.collector_jointstate[:,:,0].data.cpu().numpy(), posture_sim.collector_jointstate[:,:,1].data.cpu().numpy())
            plt.plot(input_targ_seq[-1,:,0].data.cpu().numpy(), input_targ_seq[-1,:,1].data.cpu().numpy(), 'o')
            plt.show()
            
            plt.figure()
            plt.plot(posture_sim.collector_muscleactivity[:,1,:].data.cpu().numpy())
            plt.show()
            
            plt.figure()
            plt.plot(posture_sim.collector_outplayeractivity[:, 0, :].data.cpu().numpy())
            plt.show()
            
        if epoch%100 == 0:
            outplayer_data = posture_sim.collector_outplayeractivity.data.cpu().numpy()
            file_no = 'trial' + '0' + str(num_optimizations)
            scipy.io.savemat('data/M1_' + file_name + fb_type + file_no + '.mat', {'m1data':outplayer_data})
            
            networkinp_data = posture_sim.collector_networkinputs.data.cpu().numpy()
            scipy.io.savemat('data/Inp_' + file_name + fb_type + file_no + '.mat', {'inpdata':networkinp_data})
            
            inplayer_data = posture_sim.collector_inplayeractivity.data.cpu().numpy()
            scipy.io.savemat('data/S1_' + file_name + fb_type + file_no + '.mat', {'S1data':inplayer_data})
            
            muscle_data = posture_sim.collector_muscleactivity.data.cpu().numpy()
            scipy.io.savemat('data/Mus_' + file_name + fb_type + file_no + '.mat', {'musdata':muscle_data})
            
            cartesiankin_data = posture_sim.collector_cartesianstate.data.cpu().numpy()
            scipy.io.savemat('data/Cartkin_' + file_name + fb_type + file_no + '.mat', {'cartkindata':cartesiankin_data})
            
            jointkin_data = posture_sim.collector_jointstate.data.cpu().numpy()
            scipy.io.savemat('data/Jointkin_' + file_name + fb_type + file_no + '.mat', {'jointkindata':jointkin_data})
            
            torque_data = posture_sim.collector_jointtorques.data.cpu().numpy()
            scipy.io.savemat('data/Torque_' + file_name + fb_type + file_no + '.mat', {'torquedata':torque_data})
            
            scipy.io.savemat('data/Cost_' + file_name + fb_type + file_no + '.mat', {'costcurve':cost_curve_for_plotting})
            
        if num_optimizations == 1 and epoch > 1000:
            torch.save(posture_sim.state_dict(), 'Optimizednetwork_' + file_name + fb_type + file_no + '.pth')
    
    num_optimizations += 1