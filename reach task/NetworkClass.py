import torch
import torch.nn as nn
import numpy as np
from muscularArmClass import muscular_arm
from torch.distributions import normal

'''Modifications from the manuscript simulation settings for reach task:
    To enable relatively faster optimization, time durations have been reduced 
from the original article. In the article the hold_cue/GO_signal (that indicates the arm
should stay at home even when target appears on screen) turns off at 800ms (80 time-steps), 
and total simulation runs for 300 time-steps. In this simulation, hold-cue does not
exist. The simulated arm should start moving towards the target as soon as it is
presented. Hence only step-targets signal is simulated here. '''

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class feedback_controller(nn.Module):
    def __init__(self, dh, home_joint_state):
        super(feedback_controller, self).__init__()
        # fixed control parameters
        self.dh = dh # sim-step
        self.num_network_inputs = 12
        self.num_inplayer_neurons = 500
        self.num_outplayer_neurons = 500
        self.num_muscles = 6
        self.num_reach_combinations = 17
        self.fb_delay = 5
        self.num_network_layers = 2
        self.rec_conn = 'True'
        
        self.home_joint_state = home_joint_state
        
        
        # Motor noise
        self.noise_neural = normal.Normal(0.0, 0.005)
        self.noise_muscle = normal.Normal(0.0, 0.005)
        
        
        # Intantiate the biomechanical arm dynamics at home location
        self.adyn = muscular_arm(dh)
        
        # Create neural network by means of torch functions
        # network config:
        #    1. inplayer
        #    2. outplayer
        #    3. muscle layer

        
        # Input layer receives sensory input + recurrent connections
        self.inplayer = nn.Linear(self.num_network_inputs, self.num_inplayer_neurons, bias=True)
        self.inplayer.bias = torch.nn.Parameter(torch.zeros(self.num_inplayer_neurons))
        #self.inplayer.weight = torch.nn.Parameter(nn.init.normal_(torch.empty(num_inplayer_neurons, num_network_inputs)) * self.g**2)
        self.inplayerself = nn.Linear(self.num_inplayer_neurons, self.num_inplayer_neurons, bias=True)
        self.inplayerself.bias = torch.nn.Parameter(torch.zeros(self.num_inplayer_neurons))

        # Output layer receives inputs from inplayer and recurrent inputs
        self.outplayer = nn.Linear(self.num_inplayer_neurons, self.num_outplayer_neurons, bias=True)
        self.outplayer.bias = torch.nn.Parameter(torch.zeros(self.num_outplayer_neurons))
        #self.outplayer.weight = torch.nn.Parameter(nn.init.normal_(torch.empty(num_outplayer_neurons, self.num_inplayer_neurons)) * self.g**2)
        self.outplayerself = nn.Linear(self.num_outplayer_neurons, self.num_outplayer_neurons, bias=True)
        self.outplayerself.bias = torch.nn.Parameter(torch.zeros(self.num_outplayer_neurons))
        #self.outplayerself.weight = torch.nn.Parameter(nn.init.normal_(torch.empty(num_outplayer_neurons, num_outplayer_neurons)) * self.g**2)
        
        # Muscle inputs are computed as summed contributions from outputlayer neurons
        self.musclelayer = nn.Linear(self.num_outplayer_neurons, self.num_muscles, bias=False)

        # Activation function for each network layer
        #self.musclelayer.weight = torch.nn.Parameter(nn.init.normal_(torch.empty(num_muscles, num_outplayer_neurons)) * self.g**2)
        self.inplayer_act = nn.Tanh() # activation function of layer neurons
        self.outplayer_act = nn.Tanh() # activation of layer neurons
        self.musclelayer_act = nn.ReLU() # muscle activation function only pull
        #self.musclelayer_act = nn.Tanh() # muscle activation function push-pull
        
        # initialize state information variables (joints and arm coordinates)
        self.joint_state = torch.zeros(self.num_reach_combinations, 4).to(device)
        self.cart_state = torch.zeros(self.num_reach_combinations, 4).to(device)
        self.joint_state[:, 0] = home_joint_state[0,0] # initial shoulder angle
        self.joint_state[:, 1] = home_joint_state[0,1] # initial elbow angle
        self.cart_state = self.adyn.armkin(self.joint_state)
        self.home_cart_state = self.cart_state
        self.home_joint_state = self.joint_state
        
       # set the containers for collecting simulation data
        self.collector_networkinputs = torch.empty(0, dtype=torch.float)
        self.collector_inplayeractivity = torch.empty(0, dtype=torch.float)
        self.collector_outplayeractivity = torch.empty(0, dtype=torch.float)
        self.collector_muscleactivity = torch.empty(0, dtype=torch.float)
        self.collector_jointstate = torch.empty(0, dtype=torch.float)
        self.collector_cartesianstate = torch.empty(0, dtype=torch.float)
        self.hold_data = torch.empty(0, dtype=torch.float)
        
    def forward(self, des_targ, variable_movinit_delay):
        tau_h = 0.5 # neuronal discretized leak (time constant, tau=20ms, dt/tau = 0.01/0.02 = 0.5) 
        tau_m = 0.2 # muscle activation discretized leak (time constant, tau=50ms, dt/tau = 0.01/0.05 = 0.2) 
        
        # initialize the network inputs and neuronal states
        des_pos = (des_targ[0, :, :] - self.home_joint_state[:, 0:2])
        joint_pos_fb = (self.joint_state[:, 0:2] - des_targ[0, :, :])
        joint_vel_fb = self.joint_state[:, 2:4] / 2 # usually joint-vel in the range of 0.6 rad/sec
        muscle_fb = torch.zeros(17, 6).to(device)
        network_inputs = torch.cat((des_pos, joint_pos_fb, joint_vel_fb, muscle_fb), 1)
        #network_inputs = torch.cat((hold_cmd, des_pos, joint_pos_fb, joint_vel_fb), 1) 
        
        inplayer_outputs = self.inplayer(network_inputs) 
        inplayer_outputs = tau_h*self.inplayer_act(inplayer_outputs) 
        #inplayer_outputs = torch.clamp(inplayer_outputs, min=-0.0, max=0.5)
        outplayer_outputs = self.outplayer(inplayer_outputs) 
        outplayer_outputs = tau_h*self.outplayer_act(outplayer_outputs) 
        
        #outplayer_outputs = torch.clamp(outplayer_outputs, min=-0.0, max=0.5)
        musclelayer_outputs = self.musclelayer(outplayer_outputs) 
        musclelayer_outputs = tau_m * self.musclelayer_act(musclelayer_outputs)

        
        # collect the initial states into the simulation-containers
        self.collector_networkinputs = torch.cat((self.collector_networkinputs,network_inputs.unsqueeze(0)),0)
        self.collector_inplayeractivity=torch.cat((self.collector_inplayeractivity,inplayer_outputs.unsqueeze(0)),0)
        self.collector_outplayeractivity=torch.cat((self.collector_outplayeractivity,outplayer_outputs.unsqueeze(0)),0)
        self.collector_muscleactivity=torch.cat((self.collector_muscleactivity,musclelayer_outputs.unsqueeze(0)),0)
        self.collector_jointstate = torch.cat((self.collector_jointstate,self.joint_state.unsqueeze(0)), 0)
        self.collector_cartesianstate = torch.cat((self.collector_cartesianstate,self.cart_state.unsqueeze(0)), 0)

        # start control simulation over time - BODY of the code
        # des_targ.size(0) gives the value of time duration 'T'
        for i in range(des_targ.size(0) - 1):
   
            if i > self.fb_delay and i <= 30:
                des_pos = 0*(des_targ[0, :, :] - self.home_joint_state[:, 0:2])
                joint_pos_fb = (self.collector_jointstate[i - self.fb_delay, :, 0:2] - des_targ[0, :, :])
                # or can use pure displacement information from home_location as pos_fb
                #joint_pos_fb = (self.collector_jointstate[i - self.fb_delay, :, 0:2] - self.home_joint_state[:, 0:2])
                joint_vel_fb = self.collector_jointstate[i - self.fb_delay, :, 2:4]/2
                muscle_fb = self.collector_muscleactivity[i - self.fb_delay, :, :]
            
            if i > 30:
                des_pos = (des_targ[50, :, :] - self.home_joint_state[:, 0:2])
                joint_pos_fb = (self.collector_jointstate[i - self.fb_delay, :, 0:2] - des_targ[0, :, :])
                # or can use pure displacement information from home_location as pos_fb
                #joint_pos_fb = (self.collector_jointstate[i - self.fb_delay, :, 0:2] - self.home_joint_state[:, 0:2])
                joint_vel_fb = self.collector_jointstate[i-self.fb_delay, :, 2:4]/2
                muscle_fb = self.collector_muscleactivity[i-self.fb_delay, :, :]
            
            if i < self.fb_delay:
                des_pos = (des_targ[0, :, :] - self.home_joint_state[:, 0:2])
                joint_pos_fb = (self.collector_jointstate[0, :, 0:2] - des_targ[0, :, :])
                # or can use pure displacement information from home_location as pos_fb
                #joint_pos_fb = (self.collector_jointstate[0, :, 0:2] - self.home_joint_state[:, 0:2])
                joint_vel_fb = self.collector_jointstate[0, :, 2:4]/2
                muscle_fb = self.collector_muscleactivity[0, :, :]
            
                       
            # total controller inputs 
            network_inputs = torch.cat((des_pos, joint_pos_fb, joint_vel_fb, muscle_fb), 1) 
            
            #network_inputs = torch.cat((hold_cmd, des_pos, joint_pos_fb, joint_vel_fb), 1) 
                
            # activate the network layers with network_inputs
            prev_inplayer_outputs = inplayer_outputs
            prev_outplayer_outputs = outplayer_outputs
            prev_musclelayer_outputs = musclelayer_outputs
            
            if self.num_network_layers == 1:
            
                # dynamical leaky-integrator units
                if self.rec_conn == 'True':
                    inplayer_outputs = self.inplayer(network_inputs) + self.inplayerself(prev_inplayer_outputs)
                if self.rec_conn == 'False':
                    inplayer_outputs = self.inplayer(network_inputs)
                inplayer_outputs = tau_h * self.inplayer_act(inplayer_outputs) + (1 - tau_h) * prev_inplayer_outputs 
                inplayer_outputs += self.noise_neural.sample([self.num_reach_combinations, self.num_inplayer_neurons]).to(device) * prev_inplayer_outputs*prev_inplayer_outputs
                    
                # muscle outputs 
                musclelayer_outputs = self.musclelayer(inplayer_outputs)
                musclelayer_outputs = tau_m * self.musclelayer_act(musclelayer_outputs) + (1 - tau_m) * prev_musclelayer_outputs
                musclelayer_outputs += (self.noise_muscle.sample([self.num_reach_combinations, self.num_muscles]).to(device)* prev_musclelayer_outputs*prev_musclelayer_outputs).to(device)
            
            if self.num_network_layers == 2:
                # dynamical leaky-integrator units
                if self.rec_conn == 'True':
                    inplayer_outputs = self.inplayer(network_inputs) + self.inplayerself(prev_inplayer_outputs)
                if self.rec_conn == 'False':
                    inplayer_outputs = self.inplayer(network_inputs) 
                inplayer_outputs = tau_h * self.inplayer_act(inplayer_outputs) + (1 - tau_h) * prev_inplayer_outputs
                inplayer_outputs += self.noise_neural.sample([self.num_reach_combinations, self.num_inplayer_neurons]).to(device) * prev_inplayer_outputs*prev_inplayer_outputs
                
                if self.rec_conn == 'True':
                    outplayer_outputs = self.outplayer(inplayer_outputs) + self.outplayerself(prev_outplayer_outputs)
                if self.rec_conn == 'False':
                    outplayer_outputs = self.outplayer(inplayer_outputs)
                outplayer_outputs = tau_h * self.outplayer_act(outplayer_outputs) + (1 - tau_h) * prev_outplayer_outputs 
                outplayer_outputs += (self.noise_neural.sample([self.num_reach_combinations, self.num_outplayer_neurons]).to(device)* prev_outplayer_outputs*prev_outplayer_outputs).to(device)
                
                # muscle outputs
                musclelayer_outputs = self.musclelayer(outplayer_outputs)
                musclelayer_outputs = tau_m * self.musclelayer_act(musclelayer_outputs) + (1 - tau_m) * prev_musclelayer_outputs
                musclelayer_outputs += (self.noise_muscle.sample([self.num_reach_combinations, self.num_muscles]).to(device)* prev_musclelayer_outputs*prev_musclelayer_outputs).to(device)

            # send muscle commands to the plant and get joint information
            self.joint_state, torque_output, mus_flv = self.adyn.forward(self.joint_state, musclelayer_outputs)
            
            # perform forward kinematic transformation to get arm state
            self.cart_state = self.adyn.armkin(self.joint_state)
            
            # append the current time simulation data to simulation collector variables
            self.collector_networkinputs = torch.cat((self.collector_networkinputs,network_inputs.unsqueeze(0)),0)
            self.collector_inplayeractivity = torch.cat((self.collector_inplayeractivity,inplayer_outputs.unsqueeze(0)),0)
            self.collector_outplayeractivity = torch.cat((self.collector_outplayeractivity,outplayer_outputs.unsqueeze(0)),0)
            self.collector_muscleactivity = torch.cat((self.collector_muscleactivity,musclelayer_outputs.unsqueeze(0)),0)
            self.collector_jointstate = torch.cat((self.collector_jointstate,self.joint_state.unsqueeze(0)),0)
            self.collector_cartesianstate = torch.cat((self.collector_cartesianstate,self.cart_state.unsqueeze(0)),0)
        return self.collector_jointstate
    
    def gaussianSmooth(self, t, variable_movinit_delay): # useful for generating a smooth-delayed 'GO' signal (not used in this code)
        sig = 2.5
        cur_val = (10/4) * 1/(np.sqrt(2*3.14)) * np.exp(-(t - variable_movinit_delay)**2/(2*(sig)**2))
        return cur_val
        
    def resetsim(self):  
        # state information (joints and arm coordinates)
        self.joint_state = torch.zeros(self.num_reach_combinations, 4).to(device)
        self.cart_state = torch.zeros(self.num_reach_combinations, 4).to(device)
        self.joint_state[:, 0] = self.home_joint_state[0,0] # initial shoulder angle
        self.joint_state[:, 1] = self.home_joint_state[0,1] # initial elbow angle
        self.cart_state = self.adyn.armkin(self.joint_state)
        self.home_cart_state = self.cart_state
        
        # re-set the simulation-containers for collecting simulation data
        self.collector_networkinputs = torch.empty(0, dtype=torch.float).to(device)
        self.collector_inplayeractivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_outplayeractivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_muscleactivity = torch.empty(0, dtype=torch.float).to(device)
        self.collector_jointstate = torch.empty(0, dtype=torch.float).to(device)
        self.collector_cartesianstate = torch.empty(0, dtype=torch.float).to(device)
        

def costCriterionReaching(reach_sim, actual_pos, des_pos, actual_vel, variable_movinit_delay):
    # fast reaches T=75 end duration
    # slow reaches T=110 end duration
    # Normal reaches T=80 end duration
    num_reach_combinations = actual_pos.size(1)
    num_time = actual_pos.size(0)
    # extract the actual and desired position from 500ms after the movement initiation
    # (in this case from penalize the displacement error from T=80time-steps)
    x_fT = actual_pos[80:, :]
    xd_fT = des_pos[80:, :]
    #loss = (0.05/num_time)*torch.norm((x_T - xd_T))**2 # instantaneous penalization
    loss = (0.5/50)*torch.norm((x_fT - xd_fT))**2
    # penalize for non-zero velocity 500ms after movement initiation
    loss += (0.5/50)*torch.norm(actual_vel[80:, :])**2 
    # penalize for non-zero velocity before the movement 'cue' is presented
    loss += (0.5/variable_movinit_delay)*torch.norm(actual_vel[:variable_movinit_delay, :])**2
    
    # penalize high muscle and neural activities
    loss += (1.0e-2/num_time)*(torch.norm(reach_sim.collector_muscleactivity[:variable_movinit_delay,:,:]))**2
    loss += (1.0e-4/num_time)*(torch.norm(reach_sim.collector_muscleactivity[variable_movinit_delay:,:,:]))**2
    loss += (1.0e-5/num_time)*(torch.norm(reach_sim.collector_outplayeractivity[:,:,:]))**2
    loss += (1.0e-5/num_time)*(torch.norm(reach_sim.collector_inplayeractivity[:,:,:]))**2
    #print(loss)
    # smoothness dynamics regularizer
    #loss += (1.0e-4/num_time)*(torch.norm((reach_sim.outplayerself((1 - reach_sim.collector_outplayeractivity[variable_movinit_delay:,:,:]**2))))**2)
    return (0.1/0.05)*loss/num_reach_combinations