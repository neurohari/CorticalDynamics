# -*- coding: utf-8 -*-
"""
Created on February 2021

@author: hariteja1992@gmail.com

This code optimizes the neural networks for center-out upper-limb reaching task.

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

Modifications from the manuscript simulation settings for reach task:
    To enable relatively faster optimization, time durations have been reduced 
from the original article. In the article the hold_cue/GO_signal (that indicates the arm
should stay at home even when target appears on screen) turns off at 800ms (80 time-steps), 
and total simulation runs for 300 time-steps. In this simulation, hold-cue does not
exist. The simulated arm should start moving towards the target as soon as it is
presented. Hence only a step-target signal is simulated here. 

Contact the author for simulation with additional hold-phase.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import gradcheck
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import normal
import scipy.io
from NetworkClass import feedback_controller, costCriterionReaching
import time

use_cuda = 'false'
device = torch.device('cuda:0' if use_cuda else 'cpu')
device = 'cpu'

doplot = False

# time-settings
# To enable fast optimization, time durations have been reduced from the original
# article. In the article the hold_cue turns off at 800ms (80 time-steps), and  
# total simulation runs for 300 time-steps. In this simulation, the hold-cue 
# turns-off at 300ms (30 time-steps) and total simulation duration is 1300ms.
dh = 0.01 # 10 ms sim-step
T = 1.3 # 1300ms simulation duration (130 time-steps)
#time_seq = np.linspace(0.0,T/dh-1,T/dh) # issue with float to int conversion
time_seq = np.linspace(0.0,129,130)
hold_cue_time = 30 

# Task scenario settings
num_joints = 2
num_reach_combinations = 17 # 16 radial targets + 1 home location

# arm init parameters
sho_ang = 32.6
elb_ang = 84.2
home_joint_state = torch.tensor([[np.deg2rad(sho_ang), np.deg2rad(elb_ang)]])

# large amplitude targets in joint space (5 cm in cartesian space)
xtarg = np.array([0.4368, 0.5829, 0.7279, 0.8469, 0.9182, 0.9311, 0.8892, 0.8032, 0.6840, 0.5439, 0.4017,0.2857,0.2258,0.2371,0.3138,0.4368, 0.5689])
ytarg = np.array([1.4464,1.2781,1.1311,1.0333,1.0093,1.0661,1.1878,1.3469,1.5162,1.6705,1.7863,1.8425,1.8275,1.7444,1.6098,1.4464, 1.4697])
#small amplitude target displacement in joint space (2 cm in cartesian space)
#xtarg2 = np.array([0.5102, 0.5670, 0.6239, 0.6712, 0.7011, 0.7088, 0.6938, 0.6586, 0.6087, 0.5516, 0.4969,0.4544,0.4326,0.4359,0.4636,0.5102, 0.5689])
#ytarg2 = np.array([1.4677,1.4013,1.3453,1.3098,1.3013,1.3215,1.3666,1.4282,1.4956,1.5574,1.6031,1.6249,1.6191,1.5866,1.5331,1.4677, 1.4697])
# Training mini-batch wise

reach_train_angles = np.array(np.linspace(0, 360, 16))
xtargcart = 0.05*np.cos(np.deg2rad(reach_train_angles))
ytargcart = 0.05*np.sin(np.deg2rad(reach_train_angles))
xtargcart = np.concatenate((xtargcart, np.array([0.0])), axis=0)
ytargcart = np.concatenate((ytargcart, np.array([0.0])), axis=0)
#xtargcart2 = 0.02*np.cos(np.deg2rad(reach_train_angles))
#ytargcart2 = 0.02*np.sin(np.deg2rad(reach_train_angles))
#xtargcart2 = np.concatenate((xtargcart2, np.array([0.0])), axis=0)
#ytargcart2 = np.concatenate((ytargcart2, np.array([0.0])), axis=0)

# Define the tensors to carry task information. 
# each tensor is [T, C, N].Where
# T = time sequence length
# C = number of task conditions (in this case number of different perturbations)  
# N = signal dimensionality (number of neurons (or) number of sensory-feedback components)
# input_targ_seq as network spatial target, perturb_seq as the external torques n [sho, elb] joint-directions
input_targ_seq = np.zeros([time_seq.size, num_reach_combinations, num_joints])
cart_targ_seq = np.zeros([time_seq.size, num_reach_combinations, num_joints])

### preparatory period
input_targ_seq[:30, :, 0] = 0.5689
input_targ_seq[:30, :, 1] = 1.4697
## Movement period
input_targ_seq[30:, :, 0] = xtarg[:] 
input_targ_seq[30:, :, 1] = ytarg[:] 
## preparatory period (x = x0; y0)
cart_targ_seq[:30, :, 0] = -0.0059
cart_targ_seq[:30, :, 1] = 0.3316 
# Movement period (x = r*cos(theta) - x0; y = r*sin(theta) - y0)
cart_targ_seq[30:, :, 0] = xtargcart[:] - 0.0059
cart_targ_seq[30:, :, 1] = ytargcart[:] + 0.3316


## wrap the numpy input sequence into tensors for torch
input_targ_seq = torch.from_numpy(input_targ_seq)
input_targ_seq = input_targ_seq.float().to(device)

cart_targ_seq = torch.from_numpy(cart_targ_seq)
cart_targ_seq = cart_targ_seq.float().to(device)


## simulation and file settings for training/optimizing the networks
max_simulations = 1
EPOCHS = 2000
num_optimizations = 1
file_name = 'Reach_'
fb_type = 'posvelmus_fb_'



# begin training
num_sims = 1
sim_number = '01'
# The while loop is for training multiple networks
# The for-loop inside is to train one network for fixed EPOCHS or until a reasonabe cost is obtained.
while num_optimizations <= max_simulations:
    # initialize the control class
    reach_sim = feedback_controller(dh,home_joint_state)
    reach_sim.float().to(device)
    
    # fix the network parameters 
    reach_sim.num_inplayer_neurons =500
    reach_sim.num_outplayer_neurons = 500
    reach_sim.num_muscles = 6
    reach_sim.feedback_delay = 5 # 50ms delay (or 5 time-steps)
    reach_sim.num_feedbacksignals = 12 # 2 target pos + 6 muscle_fb + 2 actual pos + 2 actual vel
    reach_sim.num_reach_combinations = num_reach_combinations
    reach_sim.num_network_layers = 2
    reach_sim.rec_conn = 'False'
    reach_sim.float()
    reach_sim.to(device)


    if reach_sim.num_network_layers == 1:
        print('Simulation using only one neural network layer...')
        if reach_sim.rec_conn == 'True':
            optim_params = list(reach_sim.inplayer.parameters()) + list(reach_sim.musclelayer.parameters()) #+ list(reach_sim.outplayer.parameters()) + list(reach_sim.m2layer.parameters()) 
            optim_params += list(reach_sim.inplayerself.parameters()) #+ list(reach_sim.outplayerself.parameters()) + list(reach_sim.m2layerself.parameters())
            print('REC connections exist')
        if reach_sim.rec_conn == 'False':
            optim_params = list(reach_sim.inplayer.parameters()) + list(reach_sim.musclelayer.parameters()) #+ list(reach_sim.outplayer.parameters()) + list(reach_sim.m2layer.parameters()) 
            print('REC connections absent')
            
    if reach_sim.num_network_layers == 2:
        print('Simulation using two neural network layers...')
        if reach_sim.rec_conn == 'True':
            optim_params = list(reach_sim.inplayer.parameters()) + list(reach_sim.musclelayer.parameters()) #+ list(reach_sim.outplayer.parameters()) + list(reach_sim.m2layer.parameters()) 
            optim_params += list(reach_sim.inplayerself.parameters()) + list(reach_sim.outplayer.parameters()) + list(reach_sim.outplayerself.parameters())
            print('REC connections exist')
        if reach_sim.rec_conn == 'False':
            optim_params = list(reach_sim.inplayer.parameters()) + list(reach_sim.musclelayer.parameters()) #+ list(reach_sim.outplayer.parameters()) + list(reach_sim.m2layer.parameters()) 
            optim_params += list(reach_sim.outplayer.parameters()) 
            print('REC connections absent')
    lrate = 2.0e-4
    optimizer = optim.Adam(optim_params, lr=lrate, weight_decay=1e-8) 
    
    # Loss value holders
    cost_curve_for_plotting = np.empty(0)
    total_loss_for_plotting = np.zeros(EPOCHS)

    # Run the optimization until the end of epochs (or can be modified to run 
    # until a satisfactory error is reached)
    t1 = time.time()
    for epoch in range(1, EPOCHS):
        # set variable preparation time-delay 
        variable_movinit_delay = np.random.randint(hold_cue_time, hold_cue_time + 1)
        input_targ_seq = np.zeros([time_seq.size, num_reach_combinations, num_joints])
        
        # target is the home location until movement init signal is presented
        input_targ_seq[:variable_movinit_delay, :, 0] = 0.5689
        input_targ_seq[:variable_movinit_delay, :, 1] = 1.4697
        # radial targets after movement init signal is presented
        input_targ_seq[variable_movinit_delay:, :, 0] = xtarg[:] 
        input_targ_seq[variable_movinit_delay:, :, 1] = ytarg[:]
        input_targ_seq = torch.from_numpy(input_targ_seq)
        input_targ_seq = input_targ_seq.float().to(device)

        reach_sim.resetsim()
        optimizer.zero_grad()
        
        joint_kinematics = reach_sim.forward(input_targ_seq, variable_movinit_delay)
        x_pos = joint_kinematics[:, :, 0]
        y_pos = joint_kinematics[:, :, 1]
        x_vel = joint_kinematics[: ,:, 2]
        y_vel = joint_kinematics[:, :, 3]

        # Compute the loss function defined by "costCriterionReaching"
        loss_x = costCriterionReaching(reach_sim, input_targ_seq[:, :, 0], x_pos, x_vel, variable_movinit_delay)
        loss_y = costCriterionReaching(reach_sim, input_targ_seq[:, :, 1], y_pos, y_vel, variable_movinit_delay)
        loss = loss_x + loss_y
        
        # Backpropagate the loss to compute loss gradients w.r.t network paramters
        loss.backward() 
        cost_curve_for_plotting = np.append(cost_curve_for_plotting, loss.item())
        
        # Update the newtork weights
        optimizer.step()
        
        total_loss_for_plotting[epoch] = loss.item()
                       
        if epoch%100 == 0:
            t2 = time.time()
            print('{:8.3f} sec for {:8f} epochs'.format(t2-t1,epoch))
            t1 = time.time()
            print("Cur instance Loss: {:.6f}".format(loss.item()), end=' ')
            print(' after Epoch number: {}/{}...'.format(epoch, EPOCHS), end=' ')
            print('num_optimizations: {}/{}'.format(num_optimizations, max_simulations, ))
            
        if (epoch%200 == 0) and doplot:
            # Plot the deisred and followed arm trajectory
            plt.figure()
            plt.plot(reach_sim.collector_cartesianstate[:,:,0].data.cpu().numpy(), reach_sim.collector_cartesianstate[:,:,1].data.cpu().numpy())
            plt.plot(cart_targ_seq[-1,:,0].data.cpu().numpy(), cart_targ_seq[-1,:,1].data.cpu().numpy(), 'o')
            plt.show()
        
            plt.figure()
            plt.plot(reach_sim.collector_jointstate[:,:,0].data.cpu().numpy(), reach_sim.collector_jointstate[:,:,1].data.cpu().numpy())
            plt.plot(input_targ_seq[-1,:,0].data.cpu().numpy(), input_targ_seq[-1,:,1].data.cpu().numpy(), 'o')
            plt.show()
            
            plt.figure()
            plt.plot(reach_sim.collector_outplayeractivity[:, 0, :].data.cpu().numpy())
            plt.show()
        
            plt.figure()
            plt.plot(reach_sim.collector_muscleactivity[:, 0, :].data.cpu().numpy())
            plt.show()
            #        
            plt.figure()
            plt.plot(reach_sim.collector_jointstate[:, 2, 0].data.cpu().numpy())
            plt.plot(input_targ_seq[:, 2, 0].data.cpu().numpy(), '--')
            plt.plot(reach_sim.collector_jointstate[:, 16, 0].data.cpu().numpy())
            plt.show()
            # 
            
        if epoch%100 == 0:
            outplayer_data = reach_sim.collector_outplayeractivity.data.cpu().numpy()
            scipy.io.savemat('data/M1_'+file_name+sim_number+'.mat', {'m1data':outplayer_data})
    
            muscle_data = reach_sim.collector_muscleactivity.data.cpu().numpy()
            scipy.io.savemat('data/Mus_'+file_name+sim_number+'.mat', {'musdata':muscle_data})
            
            cartesiankin_data = reach_sim.collector_cartesianstate.data.cpu().numpy()
            scipy.io.savemat('data/Cartkin_'+file_name+sim_number+'.mat', {'cartkindata':cartesiankin_data})
            
            jointkin_data = reach_sim.collector_jointstate.data.cpu().numpy()
            scipy.io.savemat('data/Jointkin_'+file_name+sim_number+'.mat', {'jointkindata':jointkin_data})
                
            networkinp_data = reach_sim.collector_networkinputs.data.cpu().numpy()
            scipy.io.savemat('data/Inp_'+file_name+sim_number+'.mat', {'inpdata':networkinp_data})
            
            inplayer_data = reach_sim.collector_inplayeractivity.data.cpu().numpy()
            scipy.io.savemat('data/S1_'+file_name+sim_number+'.mat', {'s1data':inplayer_data})
            
            
            torch.save(reach_sim.state_dict(), 'ReachTrainednet_'+file_name+sim_number+'.pth')
    
    num_optimizations += 1
    sim_number = '0' + str(num_sims)
            