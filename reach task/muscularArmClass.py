# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:25:40 2019

@author: Hari - hariteja1992@gmail.com
This code implements the state equations for a 2-DOF planar arm with
6 muscle actuators and non-linearities. 

In the tensor representation, the 3 dimensions represent : 
time tensor dimension - 1
sample number tensor dimension - 2
features dimension - 3 


MOST IMPORTANT NOTE - 
Do not ever perform tensor assignment in the middle of 'forward' method.
becasue torch.tensor does not remember or retain the graph unlike torch.cat.
THis results in non-computation of gradient memory.

for example the wrong code is as follows - 
h1 = ((-theta2_dot) * ((2*theta1_dot) + theta2_dot) * (self.a2 * torch.sin(theta2))) + (self.b11*theta1_dot) + (self.b12*theta2_dot)
        h2 = ((theta1_dot**2) * self.a2 * torch.sin(theta2)) + (self.b21*theta1_dot) + (self.b22*theta2_dot)
H = torch.tensor([[h1], [h2]])

above tensor assignment to H eliminates the gradient graph.

instead you should write the H assignment as 

H = torch.cat((h1.unsqueeze(0), h2.unsqueeze(0)), 0)
This retains the grad_fn during the backprop
"""
import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
device = 'cpu'

class muscular_arm():
    def __init__(self, dh=0.01):
        super(muscular_arm, self).__init__()
        
        # fixed Monkey arm parameters (0=shoulder; 1=elbow)
        self.i1 = torch.tensor([0.025]).to(device) # kg*m**2 shoulder inertia 
        self.i2 = torch.tensor([0.045]).to(device) # kg*m**2 elbow inertia
        self.m1 = torch.tensor([0.2108]).to(device) # kg mass of shopulder link
        self.m2 = torch.tensor([0.1938]).to(device) # kg mass of elbow link
        self.l1 = torch.tensor([0.145]).to(device) # meter 
        self.l2 = torch.tensor([0.284]).to(device) # meter
        self.s1 = torch.tensor([0.0749]).to(device)
        self.s2 = torch.tensor([0.0757]).to(device)
        # fixed joint-friction
       # self.b11 = torch.tensor([0.5]).to(device) # normal params
       # self.b22 = torch.tensor([0.5]).to(device) # normal params
        self.b11 = torch.tensor([0.5]).to(device) 
        self.b22 = torch.tensor([0.5]).to(device) 
        self.b21 = torch.tensor([0.1]).to(device)
        self.b12 = torch.tensor([0.1]).to(device)
        
        # inertial matrix tmp vars
        self.a1 = (self.i1 + self.i2) + (self.m2 * self.l1**2)
        self.a2 = self.m2 * self.l1 * self.s2
        self.a3 = self.i2
        
        # Moment arm param
        self.M = torch.tensor([[2.0, -2.0, 0.0, 0.0, 1.50, -2.0], [0.0, 0.0, 2.0, -2.0, 2.0, -1.50]]).to(device)
        # to remove the bi-articular muscles use
        #self.M = torch.tensor([[2.0, -2.0, 0.0, 0.0, 0.00, 0.0], [0.0, 0.0, 2.0, -2.0, 0.0, 0.0]]).to(device) 
        
        # Muscle properties
        self.theta0 = 0.0175*torch.tensor([[15.0, 4.88, 0.00, 0.00, 4.5, 2.12], [0.00, 0.00, 80.86, 109.32, 92.96, 91.52]]).to(device)
        self.L0 = torch.tensor([[7.32, 3.26, 6.4, 4.26, 5.95, 4.04]]).to(device)
        self.beta = 1.55
        self.omega = 0.81
        self.rho = 2.12
        self.Vmax = -7.39
        self.cv0 = -3.21
        self.cv1 = 4.17
        self.bv = 0.62
        self.av0 = -3.12
        self.av1 = 4.21
        self.av2 = -2.67

        # time-step of dynamics
        self.dh = dh
        
        self.cur_j_state = torch.zeros(17, 4).to(device)
        self.FV = torch.zeros(17,6).to(device)
        
    def forward(self, x, u):
        """
        rout is the readout from M1 layer. mact is the muscle activation fcn
        """
        # for linear muscle activation
        mus_inp = u
        #self.cur_j_state = x
        # for non-linear muscle activation - add F-L/V property contribution
        fl_out, fv_out = self.muscleDyn()
        flv_computed = fl_out * fv_out
        mus_out = flv_computed * mus_inp
        
        
        #muscle-force to joint-torque transformation (using M)
        self.tor = torch.mm(self.M, mus_out.transpose(0,1))

        self.tor = self.tor.transpose(0,1)

        net_command = self.tor
        out, x = self.armdyn(x, net_command)
        
        return x, net_command, mus_out
        
        
    def armdyn(self, x, u):
        batch_size = u.size(0)
        
        
        # extract joint angle states
        theta1 = x[:, 0].clone().unsqueeze(1)
        theta2 = x[:, 1].clone().unsqueeze(1)
        theta1_dot = x[:, 2].clone().unsqueeze(1)
        theta2_dot = x[:, 3].clone().unsqueeze(1)
        
    
        # compute inertia matrix
        I11 = self.a1 + (2*self.a2*(torch.cos(theta2)))
        I12 = self.a3 + (self.a2*(torch.cos(theta2)))
        I21 = self.a3 + (self.a2*(torch.cos(theta2)))
        I22 = self.a3
        I22 = I22.repeat(batch_size,1)
        

        # compute determinant of mass matrix [a * b of two tensors is the element-wise product]
        det = (I11 * I22) - (I12 * I12)

        # compute Inverse of inertia matrix
        Irow1 = torch.cat((I22, -I12), 1)
        Irow2 = torch.cat((-I21, I11), 1)

#        Iinv = (1/det.unsqueeze(1)) * torch.cat((Irow1.unsqueeze(1), Irow2.unsqueeze(1)), 1) # WORKING

        
        # compute extra torque H (coriolis, centripetal, friction)
        h1 = ((-theta2_dot) * ((2*theta1_dot) + theta2_dot) * (self.a2 * torch.sin(theta2))) + (self.b11*theta1_dot) + (self.b12*theta2_dot)
        h2 = ((theta1_dot**2) * self.a2 * torch.sin(theta2)) + (self.b21*theta1_dot) + (self.b22*theta2_dot)
        

        H = torch.cat((h1, h2), 1)

        
        # compute xdot = inv(M) * (u - H)
        #torque = u - H
        
        
        #print(torque)
        #torque = torque.unsqueeze(2) # WORKING
        #print(torque)
        # determione the terms in xdot matrix; xdot = [[dq1], [dq2], [ddq1], [ddq2]]
        dq1 = theta1_dot
        dq2 = theta2_dot
        dq = torch.cat((dq1, dq2), 1)
        
        
        
        # VISCOUS FORCE-FEILD
        #torque = u - H + ext_force
        
        torque = u - H
        
        Irow1 = (1/det) * Irow1
        Irow2 = (1/det) * Irow2
        # terms of Iinv matrix
        Iinv_11 = Irow1[:, 0].unsqueeze(1)
        Iinv_12 = Irow1[:, 1].unsqueeze(1)
        Iinv_21 = Irow2[:, 0].unsqueeze(1)
        Iinv_22 = Irow2[:, 1].unsqueeze(1)
        
        # Update acceleration of shoulder and elbow joints - FWDDYN equations        
        ddq1 = Iinv_11*torque[:, 0].unsqueeze(1) + Iinv_12*torque[:, 1].unsqueeze(1)
        ddq2 = Iinv_21*torque[:, 0].unsqueeze(1) + Iinv_22*torque[:, 1].unsqueeze(1)
        ddq = torch.cat((ddq1, ddq2), 1)
                
        
        #ddq = torch.matmul(Iinv, torque) # matmul is a bit slower than the <einsum> by 1 sec for batch matrix multiplication # WORKING
        #ddq = torch.einsum('ijk,ikl->ijl', [Iinv, torque]) # WORKING
        
        # update xdot
        x_dot = torch.cat((dq, ddq), 1)
        
        # step-update from x to x_next
        x_next = x + (self.dh * x_dot) 
        
        x = x_next
        out = x[:, 0:2]
        return out, x
    
    def armkin(self, x):
        theta1 = x[:, 0].clone().unsqueeze(1)
        theta2 = x[:, 1].clone().unsqueeze(1)
        theta1_dot = x[:, 2].clone().unsqueeze(1)
        theta2_dot = x[:, 3].clone().unsqueeze(1)
        
        g11 = (self.l1 * torch.cos(theta1)) + (self.l2 * torch.cos(theta1+theta2)) 
        g12 = (self.l1*torch.sin(theta1)) + (self.l2*torch.sin(theta1+theta2))
        g13 = -theta1_dot*((self.l1*torch.sin(theta1))+(self.l2*torch.sin(theta1+theta2)))
        g13 = g13-(theta2_dot*(self.l2*torch.sin(theta1+theta2)))
        g14 = theta1_dot*((self.l1*torch.cos(theta1))+(self.l2*torch.cos(theta1+theta2)))
        g14=g14+(theta2_dot*(self.l2*torch.cos(theta1+theta2)))
        y = torch.cat((g11,g12,g13,g14), 1)
        return y
    
    def muscleDyn(self):
        
        # F-L/V dependency
        mus_l = 1 + self.M[0,:] * (self.theta0[0,:] - self.cur_j_state[:, 0].unsqueeze(1))/self.L0 + self.M[1,:] * (self.theta0[1,:] - self.cur_j_state[:, 1].unsqueeze(1))/self.L0
        mus_v = self.M[0, :] * self.cur_j_state[:, 2].unsqueeze(1)/self.L0 + self.M[1, :] * self.cur_j_state[:, 3].unsqueeze(1)/self.L0    
        FL = torch.exp(-torch.abs((mus_l**self.beta - 1)/self.omega)**self.rho)
        FV = self.FV.clone()
        for i in range(0, 6):
            vel_i = mus_v[:, i]
            len_i = mus_l[:,i]
            FV[vel_i<=0, i] = (self.Vmax - vel_i[vel_i<=0])/(self.Vmax + vel_i[vel_i<=0]*(self.cv0 + self.cv1*len_i[vel_i<=0]))
            FV[vel_i >0, i] = (self.bv - vel_i[vel_i>0]*(self.av0+self.av1*len_i[vel_i>0]+self.av2*len_i[vel_i>0]**2))/(self.bv + vel_i[vel_i>0])

        return FL, FV
