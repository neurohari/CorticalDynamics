# cpde fpr storing motor system network classes

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from muscularArmClass import muscular_arm
from torch.distributions import normal

use_cuda = 'false'
device = torch.device('cuda:0' if use_cuda else 'cpu')
device = 'cpu'

class feedback_controller(nn.Module):
    def __init__(self, dh, home_joint_state):
        super(feedback_controller, self).__init__()
        # fixed control parameters - default values
        self.dh = dh # simulation time-step
        self.num_network_inputs = 12
        self.num_inplayer_neurons = 500
        self.num_outplayer_neurons = 500
        self.num_muscles = 6
        self.num_torque_combinations = 9
        self.rec_connection_status = 'True'
        
        self.home_joint_state = home_joint_state
        
        # Motor noise
        self.noise_neural = normal.Normal(0.0, 0.03)
        self.noise_muscle = normal.Normal(0.0, 0.03)
        
        # Intantiate the biomechanical arm dynamics at home location
        self.arm_dynamics = muscular_arm(dh)

        # Create neural network by means of torch functions
        # network config:
        #    1. inplayer
        #    2. outplayer
        #    3. muscle layer
        
        # Input layer receives sensory input + recurrent connections 
        self.inplayer = nn.Linear(self.num_network_inputs, self.num_inplayer_neurons, bias=True)
        self.inplayer.bias = torch.nn.Parameter(0.0*torch.ones(self.num_inplayer_neurons))
        # self/recurrent connections within input layer
        self.inplayerself = nn.Linear(self.num_inplayer_neurons, self.num_inplayer_neurons, bias=False)
        
        # Output layer receives inputs from inplayer and recurrent inputs
        self.outplayer = nn.Linear(self.num_inplayer_neurons, self.num_outplayer_neurons, bias=True)
        self.outplayer.bias = torch.nn.Parameter(0.0*torch.ones(self.num_outplayer_neurons))
        # self/recurrent connections within output layer
        self.outplayerself = nn.Linear(self.num_outplayer_neurons, self.num_outplayer_neurons, bias=False)
        
        # Muscle inputs are computed as summed contributions from outputlayer neurons
        self.musclelayer = nn.Linear(self.num_outplayer_neurons, self.num_muscles, bias=False)
        
        # Activation functions for neurons are Tanh()
        self.inplayer_act = nn.Tanh() # activation function of layer neurons
        self.outplayer_act = nn.Tanh() # activation of layer neurons
        
        # Muscles only produce pull-force, hence use a ReLU activation function
        self.musclelayer_act = nn.ReLU() 
        

        # initialize state information variables (joints and arm coordinates)
        self.joint_state = torch.zeros(self.num_torque_combinations, 4).to(device)
        self.cart_state = torch.zeros(self.num_torque_combinations, 4).to(device)
        self.joint_state[:, 0] = self.home_joint_state[:,0].to(device) # initial shoulder angle
        self.joint_state[:, 1] = self.home_joint_state[:,1].to(device) # initial elbow angle
        self.cart_state = self.arm_dynamics.armkin(self.joint_state)
        self.home_cart_state = self.cart_state
        self.home_joint_state = self.joint_state

        # set the containers for collecting simulation data
        self.collector_networkinputs = torch.empty(0, dtype=torch.float).to(device)
        self.collector_inplayeractivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_outplayeractivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_muscleactivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_jointstate = torch.empty(0, dtype=torch.float).to(device)
        self.collector_cartesianstate = torch.empty(0, dtype=torch.float).to(device)
        self.collector_jointtorques = torch.empty(0, dtype=torch.float).to(device)

    def forward(self, des_targ, perturb_seq):
        tau_h = 0.5 # neuronal discretized leak (time constant, tau=20ms, dt/tau = 0.01/0.02 = 0.5) 
        tau_m = 0.2 # muscle activation discretized leak (time constant, tau=50ms, dt/tau = 0.01/0.05 = 0.2) 

        # initialize the network inputs and neuronal states
        des_pos = 0*des_targ[0, :, :] 
        joint_pos_fb = (self.joint_state[:, 0:2] - self.home_joint_state[:, 0:2])*0
        joint_vel_fb = self.joint_state[:, 2:4]*0
        muscle_fb = torch.zeros(self.num_torque_combinations, 6).to(device)
        network_inputs = torch.cat((des_pos.to(device), joint_pos_fb.to(device), 0.5*joint_vel_fb.to(device), muscle_fb.to(device)), 1)
        
        inplayer_outputs = self.inplayer(network_inputs) 
        inplayer_outputs = self.inplayer_act(inplayer_outputs) 

        outplayer_outputs = self.outplayer(inplayer_outputs) #+ self.init_m1_spontaneous
        outplayer_outputs = self.outplayer_act(outplayer_outputs) 
        
        musclelayer_outputs = self.musclelayer(outplayer_outputs)
        musclelayer_outputs = self.musclelayer_act(musclelayer_outputs)
        torque_command = 0*self.joint_state[:, 2:4]

        # collect the initial states into the simulation-containers
        self.collector_networkinputs = torch.cat((self.collector_networkinputs, network_inputs.unsqueeze(0)),0)
        self.collector_inplayeractivity=torch.cat((self.collector_inplayeractivity, inplayer_outputs.unsqueeze(0)),0)
        self.collector_outplayeractivity=torch.cat((self.collector_outplayeractivity, outplayer_outputs.unsqueeze(0)),0)
        self.collector_muscleactivity=torch.cat((self.collector_muscleactivity, musclelayer_outputs.unsqueeze(0)),0)
        self.collector_jointstate = torch.cat((self.collector_jointstate, self.joint_state.unsqueeze(0)), 0)
        self.collector_cartesianstate = torch.cat((self.collector_cartesianstate, self.cartesian_state.unsqueeze(0)), 0)
        self.collector_jointtorques = torch.cat((self.collector_jointtorques, torque_command.unsqueeze(0)),0)

        # start control simulation over time - BODY of the code
        # des_targ.size(0) gives the value of time duration 'T'
        for i in range(des_targ.size(0)-1):
            # goal pos input
            des_pos = 0*des_targ[i+1, :, :]
            # feedback signals - pos and vel feedback joint coordinates
            if i >= 5: # to accomodate sensory processing delays 5 time-steps = 50ms
                jpos_fb = (self.collector_jointstate[i-5, :, 0:2]- self.home_joint_state[:, 0:2])
                jvel_fb = self.collector_jointstate[i-5, :, 2:4] 
                mus_fb = self.collector_muscleactivity[i-5, :, :]

            if i < 5:
                jpos_fb = (self.collector_jointstate[0, :, 0:2] - self.home_joint_state[:, 0:2])
                jvel_fb = self.collector_jointstate[0, :, 2:4]
                mus_fb = self.collector_muscleactivity[0, :, :]
            
            # extract perturbation(at time instance 'i') from the perturbation sequence
            cur_perturbation = perturb_seq[i, :, :]
            
            # total controller inputs
            # can either use conditioned inputs (soft-normalized) within the range of joint motion 
            # (useful in fast training of NOREC network)
            network_inputs = torch.cat((des_pos, 2*jpos_fb, 0.5*jvel_fb, mus_fb), 1) 
            # or use raw sensory data from the muscular arm from next line
            # network_inputs = torch.cat((des_pos, jpos_fb, jvel_fb, mus_fb), 1)

            # activate the network layers with network_inputs
            prev_inplayer_outputs = inplayer_outputs 
            prev_outplayer_outputs = outplayer_outputs
            prev_musclelayer_outputs = musclelayer_outputs
            
            if self.rec_connection_status == 'True':
                # dynamical leaky-integrator units
                inplayer_outputs  = self.inplayer(network_inputs) + self.inplayerself(prev_inplayer_outputs) #+ self.init_filt_spontaneous
                inplayer_outputs  = tau_h * self.inplayer_act(inplayer_outputs) + (1 - tau_h) * prev_inplayer_outputs 
                inplayer_outputs += (self.noise_neural.sample([self.num_torque_combinations, self.num_inplayer_neurons]).to(device)* prev_inplayer_outputs*prev_inplayer_outputs).to(device)
                
                outplayer_outputs = self.outplayer(inplayer_outputs) + self.outplayerself(prev_outplayer_outputs)
                outplayer_outputs = tau_h * self.outplayer_act(outplayer_outputs) + (1 - tau_h) * prev_outplayer_outputs 
                outplayer_outputs += (self.noise_neural.sample([self.num_torque_combinations, self.num_outplayer_neurons]).to(device)* prev_outplayer_outputs*prev_outplayer_outputs).to(device)
            else:
                inplayer_outputs  = self.inplayer(network_inputs) + 0*self.inplayerself(prev_inplayer_outputs) #+ self.init_filt_spontaneous
                inplayer_outputs  = tau_h * self.inplayer_act(inplayer_outputs) + (1 - tau_h) * prev_inplayer_outputs 
                inplayer_outputs += (self.noise_neural.sample([self.num_torque_combinations, self.num_inplayer_neurons]).to(device)* prev_inplayer_outputs*prev_inplayer_outputs).to(device)
                
                outplayer_outputs = self.outplayer(inplayer_outputs) + 0*self.outplayerself(prev_outplayer_outputs)
                outplayer_outputs = tau_h * self.outplayer_act(outplayer_outputs) + (1 - tau_h) * prev_outplayer_outputs 
                outplayer_outputs += (self.noise_neural.sample([self.num_torque_combinations, self.num_outplayer_neurons]).to(device)* prev_outplayer_outputs*prev_outplayer_outputs).to(device)

            # muscle outputs 
            musclelayer_outputs = self.musclelayer(outplayer_outputs) 
            musclelayer_outputs = tau_m * self.musclelayer_act(musclelayer_outputs) + (1 - tau_m) * prev_musclelayer_outputs 
            musclelayer_outputs += (self.noise_muscle.sample([self.num_torque_combinations, self.num_muscles]).to(device)* prev_musclelayer_outputs*prev_musclelayer_outputs).to(device)
            
            # send muscle commands to the plant and get joint information
            self.joint_state, torque_output, mus_flv = self.arm_dynamics.forward(self.joint_state, musclelayer_outputs, cur_perturbation)

            # perform forward kinematic transformation to get arm state
            self.cartesian_state = self.arm_dynamics.armkin(self.joint_state)
            
            # append the current time simulation data to simulation collector variables
            self.collector_networkinputs = torch.cat((self.collector_networkinputs, network_inputs.unsqueeze(0)),0)
            self.collector_inplayeractivity=torch.cat((self.collector_inplayeractivity, inplayer_outputs.unsqueeze(0)),0)
            self.collector_outplayeractivity=torch.cat((self.collector_outplayeractivity, outplayer_outputs.unsqueeze(0)),0)
            self.collector_muscleactivity=torch.cat((self.collector_muscleactivity, musclelayer_outputs.unsqueeze(0)),0)
            self.collector_jointstate = torch.cat((self.collector_jointstate, self.joint_state.unsqueeze(0)), 0)
            self.collector_cartesianstate = torch.cat((self.collector_cartesianstate, self.cartesian_state.unsqueeze(0)), 0)
            self.collector_jointtorques = torch.cat((self.collector_jointtorques, torque_command.unsqueeze(0)),0)
        return self.collector_jointstate
    
    def resetsim(self):
        # state information (joints and arm coordinates)
        self.joint_state = torch.zeros(self.num_torque_combinations, 4).to(device)
        self.cartesian_state = torch.zeros(self.num_torque_combinations, 4).to(device)
        self.joint_state[:, 0] = self.home_joint_state[:,0] # initial shoulder angle
        self.joint_state[:, 1] = self.home_joint_state[:,1] # initial elbow angle
        self.cartesian_state = self.arm_dynamics.armkin(self.joint_state)
        self.home_cartesian_state = self.cartesian_state

        # re-set the simulation-containers for collecting simulation data
        self.collector_networkinputs = torch.empty(0, dtype=torch.float).to(device)
        self.collector_inplayeractivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_outplayeractivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_muscleactivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_jointstate = torch.empty(0, dtype=torch.float).to(device)
        self.collector_cartesianstate = torch.empty(0, dtype=torch.float).to(device)
        self.collector_jointtorques = torch.empty(0, dtype=torch.float).to(device)
        
        
def costcriterionPosture(posture_sim, des_pos, actual_pos, actual_vel):
    num_torque_combinations = actual_pos.size(1)
    num_time = actual_pos.size(0)
    # to penalize displacements before the torque application (1 second)
    x_t1 = actual_pos[:100, :]
    xd_t1 = des_pos[:100, :]
    # to penalize arm not returbing within 1 second
    x_t2 = actual_pos[200:, :]
    xd_t2 = des_pos[200:, :]
    loss = (0.5 /(num_time-200))*torch.norm((x_t1 - xd_t1))**2 # displacement penalty pre-perturbation
    loss += (0.5 /(num_time-200))*torch.norm((x_t2 - xd_t2))**2 # displacement penalty post 1000ms after perturbation
    loss += (0.5 * 0.5/(num_time-200))*torch.norm(actual_vel[:100, :])**2 # velocity penalty pre-perturbation
    loss += (0.5 * 0.5/(num_time-200))*torch.norm(actual_vel[200:, :])**2 # velocity penalty post 1000ms after perturbation
    # penalize muscle and neural activities 
    # (the penalty coefficients are example among a wide-range of plausible qunatities that yield similar results)
    loss += (1.0e-2/(num_time-200))*(torch.norm(posture_sim.collector_muscleactivity[:100,:,:])**2) # penalty on high muscle activities before movement/perturbation
    loss += (1.0e-5/(num_time-200))*(torch.norm(posture_sim.collector_muscleactivity[100:200,:,:])**2) # penalty on high muscle activities during movement/perturbation
    loss += (1.0e-5/(num_time))*(torch.norm(posture_sim.collector_inplayeractivity[:,:,:])**2) # penalty on high neural activities
    loss += (1.0e-5/(num_time))*(torch.norm(posture_sim.collector_outplayeractivity[:,:,:])**2) # penalty on high neural activities
    return (0.1/0.05)*loss/num_torque_combinations