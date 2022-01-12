# wave function of single slater determinant
# using LU decomposition to compute abs(determinant)
# don't keep R_onehot


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class slater_wf(nn.Module):
    """
    slater determinant wf
    """

    def __init__(self, config, nn_wf):
        super(slater_wf, self).__init__()

        self.device = config['device']
        self.config = config
        self.para = config['parameter']
        self.L = config['L']  # Lx * Ly
        self.Lx, self.Ly = config['Lx'], config['Ly']
        self.nn_wf = nn_wf
        self.Dim = config['Dim']
        self.N = config['N']








    def get_SM(self, x, R):
        R_onehot = F.one_hot(R, num_classes = self.L).float()
        SM = self.nn_wf(x)  # slater matrix,shape = ( *, n_orbit, L, N )
        SM = torch.matmul(R_onehot, SM) # R_onehot has shape [nbatch, norb, N,L]
        SM = SM  # +  0.1*torch.eye(self.N, self.N, device=self.device)

        return SM


    def get_logdet(self, x, R):
        # x shape = nbatch, norbit, L, nloc
        # R = nbatch, norbit, N


        batch, norb, N = R.shape
        SM = self.get_SM(x,R)


        mean = torch.mean( torch.abs(SM), dim=(2,3) , keepdim=True) # mean, shape = ( batch,orb,1,1)

        SM = SM/mean
        logdet = torch.det(SM)
        sign = torch.sign(logdet).prod(dim=-1)

        logdet = torch.sum( torch.log( torch.abs( logdet ) ) , dim=(-1) )  +  N*torch.sum( ( torch.log(mean) ), dim=(1,2,3) )
        # print(logdet)
        # det = torch.det(SM*mean)
        # print(det.shape)
        # print(det)
        # det = torch.prod(det, dim=-1)
        # s = torch.sign(det)
        # ld = torch.log(torch.abs(det))
        # print(s-sign, logdet-ld )
        return logdet, sign  # get  log determinant and det sign




    def get_log_abs_det(self, x, R):
        """
        use LU, to get log(abs(det))
        :param x:
        :param R:
        :return:
        """
        #
        with torch.no_grad():
            SM = self.get_SM(x, R)
            A_LU, pivots = SM.lu()
            logdet = torch.log( torch.abs(torch.diagonal(A_LU, dim1=-2, dim2=-1)) )
            logdet = logdet.sum(dim=(-1,-2))

            # det = torch.det(SM)
            # det = torch.sum(torch.log(torch.abs(det)), dim=-1)
            # print(det - logdet, det, logdet)


        return logdet

    def get_inv(self, x, R):
        SM = self.get_SM(x,R)
        return torch.inverse(SM)

    def forward(self, x, R):
        """
        output: batch, N, L. N is the number of filled electrons
        :param x:input the fock space: (batch, n_orbit, L,n_loc)
        :param R: input of position vector (batch, norbit, N)
        :return: slater wavefunction. (batch, N, L). N is the number of filled electrons
        """
        # get the full slater matrix
        logdet, signdet = self.get_logdet(x, R)
        with torch.no_grad():
            hx = self.get_local_hx(x, R, logdet,signdet)

        return logdet, signdet, hx



    def get_local_hx(self, x, R, logdet, signdet):
        """
        local energy fun for Hubbard model
        :param self:
        :param x: fock space (batch, n_orbit, L,n_loc)
        :param R: possition mat (batch, norbit,N)
        :param logdet: log det of x
        :return:
        """
        if(self.Dim == 2):
            return self.get_local_hx_2d(x, R, logdet, signdet)
        else:
            return None # to be done self.get_local_hx_1d(x, R, logdet, signdet)


    # computer detterminant parallel
    def get_local_hx_2d(self, x, R, logdet, signdet):
        """
        local energy fun for Hubbard model
        :param self:
        :param x: fock space (batch, n_orbit, L,n_loc)
        :param R: possition mat (batch, norbit,N)
        :param logdet: log det of x
        :return:
        """

        nbatch, norbit, L, nloc = x.shape


        t = self.para['t']
        u = self.para['u']
        n0 = self.para['n0']

        re_hx = torch.zeros(nbatch, device=self.device)
        im_hx = torch.zeros(nbatch, device=self.device)
        #

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:

            ix = (R // self.Ly)
            iy = (R % self.Ly)
            R2 = ((ix + dx + self.Lx) % self.Lx) * self.Ly + (iy + dy + self.Ly) % self.Ly
            index = torch.gather(x[:,:,:,0],dim=2, index=R2)  > 0.5 # shape = [nbatch, norb, N_ele], permit hopping

            for i_orb in range(norbit):
                for i_r in range(0,self.N):
                    ind = torch.nonzero(index[:, i_orb, i_r], as_tuple=False).view(-1)
                    if( len(ind) < 1) :
                        continue

                    r  = R[:,i_orb, i_r].long()  # position of i_orbital and i_r electron. shape = [batch,]
                    r2 = R2[:,i_orb, i_r].long()  # shape [batch,]


                    x_new = x[ind].clone()
                    R_new = R[ind].clone()

                    # R_onehot_new = R_onehot[ind].clone()
                    for i,i_batch in enumerate( ind ):
                        x_new[i, i_orb, r[i_batch], :], x_new[i, i_orb, r2[i_batch], :] \
                            = torch.tensor([1.0,0.0]), torch.tensor([0.0,1.0])
                        R_new[i,i_orb, i_r] = r2[i_batch]


                    with torch.no_grad():
                        logdet_new, signdet_new = self.get_logdet(x_new, R_new)

                    re_hx[ind] += torch.exp(logdet_new - logdet[ind]) * signdet_new * signdet[ind] * t





        fill = torch.sum(x[:, :, :, 1],dim=1)  # filling at each site, shape = nbatch, L
        fill = fill - n0

        re_hx += torch.sum(fill * fill / 2.0 * u, dim=1)  # hubbard int = 0.5*U(n-n0)^2


        return re_hx.detach() , im_hx.detach()
    #
    # def get_local_hx_1d(self, x, R, logdet, signdet):
    #     """
    #     local energy fun for Hubbard model
    #     :param self:
    #     :param x: fock space (batch, n_orbit, L,n_loc)
    #     :param R_onehot: one hot position mat (batch, n_orbit, L,N)
    #     :param R: possition mat (batch, norbit,N)
    #     :param logdet: log det of x
    #     :return:
    #     """
    #
    #
    #     nbatch, norbit, L, nloc = x.shape
    #     _, _, _, N = R_onehot.shape
    #
    #     t = self.para['t']
    #     u = self.para['u']
    #     n0 = self.para['n0']
    #
    #     re_hx = torch.zeros(nbatch, device=self.device)
    #     im_hx = torch.zeros(nbatch, device=self.device)
    #
    #     for i_batch in range(nbatch):  # each batch
    #         x_new = torch.clone(x[i_batch, :, :, :]).view(1, norbit, L, nloc)
    #         R_onehot_new = torch.clone(R_onehot[i_batch, :, :, :]).view(1, norbit, L, N)
    #         for i_orb in range(norbit):  # each orbit
    #             for i_r, r in enumerate(R[i_batch, i_orb, :]):  # each electron
    #                 for dx in [1,-1]:
    #                     r = r.long()
    #                     r2 = (r+dx + self.L)%self.L
    #
    #
    #                     if (torch.sum(torch.abs(x[i_batch, i_orb, r, :] - x[i_batch, i_orb, r2, :])) > 0.1):
    #                         # non-zero contribution
    #
    #                         # new config after hopping
    #                         x_new[0, i_orb, r, :], x_new[0, i_orb, r2, :] = \
    #                             x_new[0, i_orb, r2, :].clone(), x_new[0,i_orb, r,:].clone()
    #
    #                         R_onehot_new[0, i_orb, r, i_r], R_onehot_new[0, i_orb, r2, i_r] = 0.0, 1.0
    #
    #                         logdet_new, signdet_new = self.get_logdet(x_new, R_onehot_new)
    #                         signdet_new = signdet_new
    #
    #                         re_hx[i_batch] += torch.exp( logdet_new[0] - logdet[i_batch] )* signdet_new[0] * signdet[i_batch] * t
    #
    #                         # go back to the old config
    #                         x_new[0, i_orb, r, :], x_new[0, i_orb, r2, :] \
    #                             = x_new[0, i_orb, r2, :].clone(), x_new[0,i_orb, r,:].clone()
    #
    #                         R_onehot_new[0, i_orb, r, i_r], R_onehot_new[0, i_orb, r2, i_r] = 1.0, 0.0
    #
    #     fill = torch.sum(x[:, :, :, 1],dim=1)  # filling at each site, shape = nbatch, L
    #     fill = fill - n0
    #     re_hx += torch.sum(fill * fill / 2.0 * u, dim=1)  # hubbard int = 0.5*U(n-n0)^2
    #
    #     return re_hx, im_hx
