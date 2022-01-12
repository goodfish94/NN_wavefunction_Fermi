# NN wavefunction ansatz with translational symmetry
# Use 1. Convolution layer; 2. Attention mechanism (see graph network)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def getPbcTensor_1d(Lx,  filter_size, stride, device):
    """
    generate a tensor with dimension (Lx,Lx+del_x)
    del_x=del_y = filter_size - stride
    map (_,_,Lx) tensor to (_,_,Lx + del_x) tensor with PBC
    """
    del_x = filter_size- stride
    trans_tensor = torch.eye(Lx , (Lx + del_x) , device=device)
    for i in range(0, del_x):
            trans_tensor[i,  i + Lx] = 1.0
    return trans_tensor


class convPbc_1d(nn.Module):
    """
    conv with translational inv
    """
    def __init__(self, config, L, in_channel, out_channel, filter_size, stride):
        # out channel has size.  L' = L /stride, requires L%stride == 0
        super(convPbc_1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, filter_size, stride=stride, padding=0)
        self.L = L
        self.PbcTensor = getPbcTensor_1d(L, filter_size, stride, config['device']) # Pbc tensor with shape: [Lx, Lx+dx]

    def forward(self, x):
        # input has shape (_, inchannel, L)
        x = torch.tensordot(x, self.PbcTensor, dims=1)  # pbc
        x = self.conv(x)  # conv layer
        # output has shape (_, outchannel, L')
        # L' = L /stride, requires L%stride == 0
        return x




class TI_wf_1d(nn.Module):
    def __init__(self, config):
        super(TI_wf_1d, self).__init__()

        self.device = config['device']
        self.config = config
        self.para = config['parameter']
        self.L = config['L']  #
        self.n_orb = config['n_orb']
        self.N = config['N']

        self.pos_vec = torch.zeros(self.L, 1)  # position vector (r,)
        for i in range(0, self.L):
            self.pos_vec[i, 0] = i
        self.pos_vec = self.pos_vec.to(self.device)

        kl1 = nn.Linear(1, 20)
        nn.init.xavier_normal_(kl1.weight)
        kl2 = nn.Linear(20, 30)
        nn.init.xavier_normal_(kl2.weight)
        kl3 = nn.Linear(30, self.n_orb * int(self.N / 2))
        nn.init.xavier_normal_(kl3.weight)
        klayer = [kl1, nn.LeakyReLU(), kl2, nn.LeakyReLU(), kl3]
        self.k_layer = nn.Sequential(*klayer)

        #         self.k_layer = nn.Linear(1, self.n_orb * int(self.N/2), bias=False) # weight[0] = k

        #         nn.init.uniform_(self.k_layer.weight, a= -1.0, b=1.0 )

        self.channel = [10, 30, 50, 100, 50, 30]

        conv_list = []
        c_in = 2 * self.n_orb
        for i, c in enumerate(self.channel):
            conv_list.append(convPbc_1d(config, self.L, c_in, self.channel[i], filter_size=2, stride=1))
            c_in = self.channel[i]
            nn.init.xavier_normal_(conv_list[-1].conv.weight)
            conv_list.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*conv_list)
        self.convfinal = convPbc_1d(config, self.L, self.channel[-1], self.n_orb, filter_size=2, stride=1)
        nn.init.xavier_normal_(self.convfinal.conv.weight)

        self.fc = nn.Linear(1, self.N * self.n_orb * self.L, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.fc2 = nn.Linear(self.n_orb * self.L, self.n_orb * self.L * self.N)
        nn.init.xavier_normal_(self.fc2.weight)
        self.pseudo_input = torch.ones((1,), requires_grad=False).to(self.device)



    def get_k(self):
        with torch.no_grad():
            k = self.k_layer(self.pseudo_input).view(1, self.n_orb * int(self.N / 2))
        k = k.view(self.n_orb, int(self.N / 2)).detach()
        return k

    def get_conv_out(self, x):
        # input shape : batch, norb, L, 2

        batch, _, _, _ = x.shape
        y = torch.transpose(x, dim0=2, dim1=3)
        y = y.reshape(batch, self.n_orb * 2, self.L)
        y = self.conv(y)
        y = (self.convfinal(y))  # output has shape, [batch, norb, L]

        y = y.view(batch, self.n_orb, self.L, 1)

        return y

    def get_ft_matrix(self, pos_vec = None):
        if (pos_vec is None):
            pos_vec = self.pos_vec

        k = self.k_layer(self.pseudo_input).view(1, self.n_orb * int(self.N / 2))
        kr = (k * pos_vec).view(self.L, self.n_orb, int(self.N / 2))
        kr = torch.transpose(kr, dim0=0, dim1=1)  # shape n_orb, L, N/2
        ckr = torch.cos(kr)
        skr = torch.sin(kr)

        # cos(kr), sin(kr)
        FT = torch.cat((ckr, skr), dim=2)  # fourier transformation tensor, (n_orb, L,N)
        FT = FT[:, :, 0:self.N]
        return FT

    def forward(self, x, pos_vec=None):
        # input shape : batch, norb, L, 2
        # output slater Matrix: shape: 2, N_electron, L

        if (pos_vec is None):
            pos_vec = self.pos_vec
        batch, _, _, _ = x.shape
        y = torch.transpose(x, dim0=2, dim1=3)
        y = y.reshape(batch, self.n_orb * 2, self.L)
        y = self.conv(y)
        y = (self.convfinal(y))  # output has shape, [batch, norb, L]

        y = y.view(batch, self.n_orb, self.L, 1)


        use_ft = True
        if (use_ft):
            k = self.k_layer(self.pseudo_input).view(1, self.n_orb * int(self.N / 2))
            k = self.test_k
            k = 2.0*np.pi/self.L * k

            kr = (k * pos_vec).view(self.L, self.n_orb, int(self.N / 2))
            kr = torch.transpose(kr, dim0=0, dim1=1)  # shape n_orb, L, N/2
            ckr = torch.cos(kr)
            skr = torch.sin(kr)

            # cos(kr), sin(kr)
            FT = torch.cat((ckr, skr), dim=2)  # fourier transformation tensor, (n_orb, L,N)
            FT = FT[:, :, 0:self.N]
        else:
            FT = self.fc(self.pseudo_input).view(self.n_orb, self.L, self.N)
        y = torch.transpose(y * FT, dim0=2, dim1=3)  # shape = batch, norb, N_ele, L

        return y






def getPbcTensor(Lx, Ly, filter_size, stride, device):
    """
    generate a tensor with dimension (Lx,Ly, Lx+del_x, Ly+del_y)
    del_x=del_y = filter_size - stride
    map (_,_,Lx,Ly) tensor to (_,_,Lx + del_x, Ly + del_y) tensor with PBC
    """
    del_x, del_y = filter_size - stride, filter_size - stride
    trans_tensor = torch.eye(Lx * Ly, (Lx + del_x) * (Ly + del_y), device=device)
    trans_tensor = torch.reshape(trans_tensor, (Lx, Ly, Lx + del_x, Ly + del_y))
    for i in range(0, del_x):
        for j in range(0, del_y):
            trans_tensor[i, j, i + Lx, j + Ly] = 1.0

    return trans_tensor


class convPbc(nn.Module):
    """
    conv with translational inv
    """
    def __init__(self, config, Lx, Ly, in_channel, out_channel, filter_size, stride):
        # out channel has size.  L' = L /stride, requires L%stride == 0
        super(convPbc, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride=stride, padding=0)
        self.Lx, self.Ly = Lx, Ly
        self.PbcTensor = getPbcTensor(Lx, Ly, filter_size, stride, config['device'])

    def forward(self, x):
        # input has shape (_, inchannel, Lx,Ly)
        x = torch.tensordot(x, self.PbcTensor, dims=2)  # pbc
        x = self.conv(x)  # conv layer
        # output has shape (_, outchannel, Lx',Ly')
        # L' = L /stride, requires L%stride == 0
        return x

class TI_wf(nn.Module): # translational invriant wf, use attention mechanism or pbc conv layer
    def __init__(self, config):
        super(TI_wf, self).__init__()

        self.device = config['device']
        self.config = config
        self.para = config['parameter']
        self.L = config['L']  # Lx * Ly
        self.n_orb = config['n_orb']
        self.N = config['N']
        self.Lx, self.Ly = config['Lx'], config['Ly']


        self.channel = [10, 50, 100, 50, 30]
        self.final_channel = 10

        conv_list = []
        c_in = 2 * self.n_orb
        for i, c in enumerate(self.channel):
            conv_list.append(convPbc(config, self.Lx, self.Ly, c_in, self.channel[i], filter_size=2, stride=1))
            c_in = self.channel[i]
            nn.init.xavier_normal_(conv_list[-1].conv.weight)
            conv_list.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*conv_list)
        self.convfinal = convPbc(config, self.Lx, self.Ly, self.channel[-1], self.final_channel, filter_size=2, stride=1)
        nn.init.xavier_normal_(self.convfinal.conv.weight)



        self.fc = nn.Linear(self.L*self.final_channel, self.N * self.n_orb * self.L, bias=False)
        nn.init.xavier_normal_(self.fc.weight)


    def forward(self,x):
        # input shape : batch, norb, L, 2
        # output slater Matrix: shape: 2, N_electron, L

        batch, _, _, _ = x.shape
        y = torch.transpose(x, dim0=2, dim1=3)
        y = y.reshape(batch, self.n_orb * 2, self.Lx, self.Ly)
        y = self.conv(y)
        y = self.convfinal(y)

        y = self.fc(y).view(batch, self.n_orb, self.N,self.L)

        return y

