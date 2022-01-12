# Monte Carlo sampling part
import torch
import time

from utility import *


class Sampler:
    def __init__(self,config):
        self.L = config['L']
        self.device = config['device']
        self.dim = config['Dim']
        self.Lx, self.Ly = config['Lx'],config['Ly']

    def one_step(self, x,  R,log_abs_det, wf):

        # ts = time.time()

        # propose update
        p = torch.rand(1)
        if( p< 0.3):
            self.hopping_nhb(x,R)
        elif(p < 0.7):
            self.hopping(x,R)
        else:
            self.spin_flipping(x,R)


        if (len(self.index_) == 0):
            return x, R, log_abs_det # no permit update


        #
        # accept = torch.zeros(nbatch).to(self.device)
        # log_abs_det_new = torch.zeros(nbatch).to(self.device)
        with torch.no_grad():
            log_abs_det_new = wf.get_log_abs_det( self.xnew, self.R_new)

        accept = ( torch.exp(2.0*(log_abs_det_new-log_abs_det[self.index_]))
                                > (torch.rand(len(self.index_)).to(self.device)))

        not_accept = torch.logical_not( accept )
        log_abs_det[self.index_] = log_abs_det[self.index_] * not_accept + log_abs_det_new * accept

        accept = accept.view(-1, 1, 1)
        not_accept = not_accept.view(-1, 1, 1)
        R[self.index_] = R[self.index_] * not_accept + self.R_new * accept

        accept = accept.view(-1, 1, 1, 1)
        not_accept = not_accept.view(-1, 1, 1, 1)
        # R_onehot[self.index_] = R_onehot[self.index_] * not_accept +  self.R_onehot_new * accept
        x[self.index_] = x[self.index_] * not_accept +  self.xnew * accept

        # print("t = ", time.time() - ts)

        #
        # check_consistent(x, R)
        # SM = wf.get_SM(x,R)
        # logdet_gt = torch.log( torch.abs( torch.det(SM) ) ).sum(dim=-1)
        # print(log_abs_det - logdet_gt )
        #
        #
        # print("ind",len(self.index_))
        # print("accp",torch.mean(accept.float()))
        # print("mean log new",torch.mean(log_abs_det_new))
        # print("mean log", torch.mean(log_abs_det) )
        return x, R, log_abs_det




    def hopping(self, x,R):
        """
        propose update
        randomly hopping for each orbit
        :param x: fock space (batch, n_orbit, L,n_loc)
        :param R: possition mat (batch, norbit,N)
        :param wf: slater wf
        :return:
        """
        nbatch, norbit, L, nloc = x.shape
        N_ele = R.shape[-1]

        i_orb = (torch.randint(0,norbit,(1,),dtype=torch.long)[0]) # pick one orbit
        i_r = torch.randint( 0,N_ele, size= (nbatch,1),dtype=torch.long ).to(self.device)
        R2 = torch.randint(0,L,size=(nbatch,1),dtype=torch.long ).to(self.device)
        R1 = torch.gather( R[:,i_orb, :], dim=1, index=i_r ) # shape [nbatch,1]

        self.index_ = torch.nonzero( (torch.gather(x[:,i_orb, :,0], dim=1, index = R2) > 0.5).view(-1) , as_tuple=False).view(-1)

        if( len(self.index_) < 1):
            return

        self.xnew = torch.clone(x[self.index_])
        self.R_new = torch.clone(R[self.index_])


        for i, i_batch in enumerate( self.index_ ):

            self.xnew[i, i_orb, R1[i_batch,0], :], self.xnew[i, i_orb, R2[i_batch,0], :] = \
                torch.tensor([1.0,0.0]), torch.tensor([0.0,1.0] )
            self.R_new[i, i_orb, i_r[i_batch]] = R2[i_batch,0]



    def hopping_nhb(self, x,  R):
        """
        propose update
        hopping around 8 nhb for 2d, 2 nhbs for 1d
        randomly hopping for each orbit
        :param x: fock space (batch, n_orbit, L,n_loc)
        :param R_onehot: one hot position mat (batch, n_orbit, L,N)
        :param R: possition mat (batch, norbit,N)
        :param wf: slater wf
        :return:
        """

        nbatch, norbit, L, nloc = x.shape
        N_ele = R.shape[-1]

        i_orb = (torch.randint(0, norbit, (1,), dtype=torch.long)[0])  # pick one orbit
        i_r = torch.randint(0, N_ele, size=(nbatch, 1), dtype=torch.long).to(self.device)
        R1 = torch.gather(R[:, i_orb, :], dim=1, index=i_r)  # shape [nbatch,1]



        if (self.dim == 1):
            dx = torch.randint(-1, 1, size=[nbatch]).to(self.device)
        else:
            dx = torch.randint(-1, 2, size=[nbatch,1]).to(self.device)
            dy = torch.randint(-1, 2, size=[nbatch,1]).to(self.device)

            ix = (R1 // self.Ly).long()
            iy = (R1 % self.Ly).long()
            R2 = ((ix + dx + self.Lx) % self.Lx) * self.Ly + (iy + dy + self.Ly) % self.Ly


        self.index_ = torch.nonzero((torch.gather(x[:, i_orb, :, 0], dim=1, index=R2) > 0.5).view(-1),
                                    as_tuple=False).view(-1)

        if( len(self.index_) < 1):
            return
        self.xnew = torch.clone(x[self.index_])
        self.R_new = torch.clone(R[self.index_])


        for i, i_batch in enumerate(self.index_):

            self.xnew[i, i_orb, R1[i_batch, 0], :], self.xnew[i, i_orb, R2[i_batch, 0], :] = \
                torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
            self.R_new[i, i_orb, i_r[i_batch]] = R2[i_batch, 0]







    def spin_flipping(self,x,R):
        """
        propose update
        Exchange pos of up electron and dn electron
        randomly hopping for each orbit
        :param x: fock space (batch, n_orbit, L,n_loc)
        :param R_onehot: one hot position mat (batch, n_orbit, L,N)
        :param R: possition mat (batch, norbit,N)
        :param wf: slater wf
        :return:
        """
        orb_up = 0 # up orbital
        orb_dn = 1 # dn orbital

        nbatch, norbit, L, nloc = x.shape
        N_ele = R.shape[-1]

        p_pos_up = torch.randint(0, N_ele, size=[nbatch,1]).to(self.device)
        p_pos_dn = torch.randint(0, N_ele, size=[nbatch,1]).to(self.device)
        R_up = torch.gather(R[:,orb_up,:], dim=1, index=p_pos_up)
        R_dn = torch.gather(R[:,orb_dn,:], dim=1, index=p_pos_dn)

        n_dn_at_r_up = torch.gather(x[:,orb_dn,:,1], dim=1, index=R_up).view(-1)
        n_up_at_r_dn = torch.gather(x[:,orb_up,:,1], dim=1, index=R_dn).view(-1)
        self.index_ = torch.logical_and( n_dn_at_r_up < 0.5, n_up_at_r_dn < 0.5).view(-1) # shape = [batch,], which batch is permitted
        self.index_ = torch.nonzero(self.index_, as_tuple=False).view(-1)


        if( len(self.index_) < 1):
            return

        self.xnew = torch.clone(x[self.index_])
        self.R_new = torch.clone(R[self.index_])


        for i,i_batch in enumerate(self.index_):
            self.xnew[i, orb_up, R_up[i_batch, 0], :], self.xnew[i, orb_up, R_dn[i_batch, 0], :] = \
                torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
            self.xnew[i, orb_dn, R_up[i_batch, 0], :], self.xnew[i, orb_dn, R_dn[i_batch, 0], :] = \
                torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])

            self.R_new[i, orb_up, p_pos_up[i_batch,0]] = R_dn[i_batch,0]
            self.R_new[i, orb_dn, p_pos_dn[i_batch,0]] = R_up[i_batch,0]





    # def one_step(self, x, R_onehot, R,log_abs_det, wf):
    #
    #     ts = time.time()
    #
    #     # propose update
    #     p = torch.rand(1)
    #     if( p< 0.3):
    #         self.hopping_nhb(x,R_onehot,R, wf)
    #     elif(p < 0.7):
    #         self.hopping(x,R_onehot,R, wf)
    #     else:
    #         self.spin_flipping(x,R_onehot,R, wf)
    #
    #
    #     if (len(self.index_) == 0):
    #         return x, R_onehot, R, log_abs_det # no permit update
    #
    #
    #     #
    #     # accept = torch.zeros(nbatch).to(self.device)
    #     # log_abs_det_new = torch.zeros(nbatch).to(self.device)
    #     with torch.no_grad():
    #         log_abs_det_new = wf.get_log_abs_det( self.xnew, self.R_onehot_new)
    #
    #     accept = ( torch.exp(2.0*(log_abs_det_new-log_abs_det[self.index_]))
    #                             > (torch.rand(len(self.index_)).to(self.device)))
    #
    #     not_accept = torch.logical_not( accept )
    #     log_abs_det[self.index_] = log_abs_det[self.index_] * not_accept + log_abs_det_new * accept
    #
    #     accept = accept.view(-1, 1, 1)
    #     not_accept = not_accept.view(-1, 1, 1)
    #     R[self.index_] = R[self.index_] * not_accept + self.R_new * accept
    #
    #     accept = accept.view(-1, 1, 1, 1)
    #     not_accept = not_accept.view(-1, 1, 1, 1)
    #     R_onehot[self.index_] = R_onehot[self.index_] * not_accept +  self.R_onehot_new * accept
    #     x[self.index_] = x[self.index_] * not_accept +  self.xnew * accept
    #
    #     print("t = ", time.time() - ts)
    #     return x, R_onehot, R, log_abs_det
    #
    #

    #
    # def hopping(self, x,R_onehot,R,wf):
    #     """
    #     propose update
    #     randomly hopping for each orbit
    #     :param x: fock space (batch, n_orbit, L,n_loc)
    #     :param R_onehot: one hot position mat (batch, n_orbit, L,N)
    #     :param R: possition mat (batch, norbit,N)
    #     :param wf: slater wf
    #     :return:
    #     """
    #     nbatch, norbit, L, nloc = x.shape
    #     _,_,_, N_ele = R_onehot.shape
    #
    #     i_orb = (torch.randint(0,norbit,(1,),dtype=torch.long)[0]).to(self.device) # pick one orbit
    #     i_r = torch.randint( 0,N_ele, size= (nbatch,1),dtype=torch.long ).to(self.device)
    #     R2 = torch.randint(0,L,size=(nbatch,1),dtype=torch.long ).to(self.device)
    #     R1 = torch.gather( R[:,i_orb, :], dim=1, index=i_r ) # shape [nbatch,1]
    #
    #     self.index_ = torch.nonzero( (torch.gather(x[:,i_orb, :,0], dim=1, index = R2) > 0.5).view(-1) , as_tuple=False).view(-1)
    #
    #
    #     self.xnew = torch.clone(x[self.index_])
    #     self.R_onehot_new = torch.clone(R_onehot[self.index_])
    #     self.R_new = torch.clone(R[self.index_])
    #
    #
    #     for i, i_batch in enumerate( self.index_ ):
    #
    #         self.xnew[i, i_orb, R1[i_batch,0], :], self.xnew[i, i_orb, R2[i_batch,0], :] = \
    #             torch.tensor([1.0,0.0]), torch.tensor([0.0,1.0] )
    #         self.R_onehot_new[i, i_orb, R1[i_batch,0], i_r[i_batch]], self.R_onehot_new[i, i_orb, R2[i_batch,0], i_r[i_batch]] = 0.0, 1.0
    #         self.R_new[i, i_orb, i_r[i_batch]] = R2[i_batch,0]
    #
    #
    #
    # def hopping_nhb(self, x, R_onehot, R,  wf):
    #     """
    #     propose update
    #     hopping around 8 nhb for 2d, 2 nhbs for 1d
    #     randomly hopping for each orbit
    #     :param x: fock space (batch, n_orbit, L,n_loc)
    #     :param R_onehot: one hot position mat (batch, n_orbit, L,N)
    #     :param R: possition mat (batch, norbit,N)
    #     :param wf: slater wf
    #     :return:
    #     """
    #
    #     nbatch, norbit, L, nloc = x.shape
    #     _, _, _, N_ele = R_onehot.shape
    #
    #     i_orb = (torch.randint(0, norbit, (1,), dtype=torch.long)[0])  # pick one orbit
    #     i_r = torch.randint(0, N_ele, size=(nbatch, 1), dtype=torch.long).to(self.device)
    #
    #     R1 = torch.gather(R[:, i_orb, :], dim=1, index=i_r)  # shape [nbatch,1]
    #
    #
    #
    #     if (self.dim == 1):
    #         dx = torch.randint(-1, 1, size=[nbatch]).to(self.device)
    #     else:
    #         dx = torch.randint(-1, 2, size=[nbatch,1]).to(self.device)
    #         dy = torch.randint(-1, 2, size=[nbatch,1]).to(self.device)
    #
    #         ix = (R1 // self.Ly).long()
    #         iy = (R1 % self.Ly).long()
    #         R2 = ((ix + dx + self.Lx) % self.Lx) * self.Ly + (iy + dy + self.Ly) % self.Ly
    #
    #
    #     self.index_ = (torch.gather(x[:,i_orb,:,0], dim=1, index=R2) > 0.5).view(-1)
    #     self.index_ = torch.nonzero(self.index_, as_tuple=False).view(-1)
    #
    #     self.index_ = torch.nonzero((torch.gather(x[:, i_orb, :, 0], dim=1, index=R2) > 0.5).view(-1),
    #                                 as_tuple=False).view(-1)
    #
    #     if( len(self.index_) < 1):
    #         return
    #     self.xnew = torch.clone(x[self.index_])
    #     self.R_onehot_new = torch.clone(R_onehot[self.index_])
    #     self.R_new = torch.clone(R[self.index_])
    #
    #
    #     for i, i_batch in enumerate(self.index_):
    #
    #         self.xnew[i, i_orb, R1[i_batch, 0], :], self.xnew[i, i_orb, R2[i_batch, 0], :] = \
    #             torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    #         self.R_onehot_new[i, i_orb, R1[i_batch, 0], i_r[i_batch]], self.R_onehot_new[
    #             i, i_orb, R2[i_batch, 0], i_r[i_batch]] = 0.0, 1.0
    #         self.R_new[i, i_orb, i_r[i_batch]] = R2[i_batch, 0]
    #
    #
    #
    #
    #
    #
    #
    # def spin_flipping(self,x,R_onehot,R,  wf):
    #     """
    #     propose update
    #     Exchange pos of up electron and dn electron
    #     randomly hopping for each orbit
    #     :param x: fock space (batch, n_orbit, L,n_loc)
    #     :param R_onehot: one hot position mat (batch, n_orbit, L,N)
    #     :param R: possition mat (batch, norbit,N)
    #     :param wf: slater wf
    #     :return:
    #     """
    #     orb_up = 0 # up orbital
    #     orb_dn = 1 # dn orbital
    #
    #     # STOP HERE
    #     nbatch, norbit, L, nloc = x.shape
    #     _, _, _, N_ele = R_onehot.shape
    #
    #     p_pos_up = torch.randint(0, N_ele, size=[nbatch,1]).to(self.device)
    #     p_pos_dn = torch.randint(0, N_ele, size=[nbatch,1]).to(self.device)
    #     R_up = torch.gather(R[:,0,:], dim=1, index=p_pos_up)
    #     R_dn = torch.gather(R[:,1,:], dim=1, index=p_pos_dn)
    #
    #     n_dn_at_r_up = torch.gather(x[:,1,:,1], dim=1, index=R_up).view(-1)
    #     n_up_at_r_dn = torch.gather(x[:,0,:,1], dim=1, index=R_dn).view(-1)
    #     self.index_ = torch.logical_and( n_dn_at_r_up < 0.5, n_up_at_r_dn < 0.5).view(-1) # shape = [batch,], which batch is permitted
    #     self.index_ = torch.nonzero(self.index_, as_tuple=False).view(-1)
    #
    #     self.xnew = torch.clone(x[self.index_])
    #     self.R_onehot_new = torch.clone(R_onehot[self.index_])
    #     self.R_new = torch.clone(R[self.index_])
    #
    #
    #     for i,i_batch in enumerate(self.index_):
    #         self.xnew[i, orb_up, R_up[i_batch, 0], :], self.xnew[i, orb_up, R_dn[i_batch, 0], :] = \
    #             torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    #         self.xnew[i, orb_dn, R_up[i_batch, 0], :], self.xnew[i, orb_dn, R_dn[i_batch, 0], :] = \
    #             torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    #
    #         self.R_new[i, orb_up, p_pos_up[i_batch,0]] = R_dn[i_batch,0]
    #         self.R_new[i, orb_dn, p_pos_dn[i_batch,0]] = R_up[i_batch,0]
    #
    #         self.R_onehot_new[i, orb_up, R_up[i_batch, 0], p_pos_up[i_batch, 0]] = 0.0
    #         self.R_onehot_new[i, orb_up, R_dn[i_batch, 0], p_pos_up[i_batch, 0]] = 1.0
    #         self.R_onehot_new[i, orb_dn, R_dn[i_batch, 0], p_pos_dn[i_batch, 0]] = 0.0
    #         self.R_onehot_new[i, orb_dn, R_dn[i_batch, 0], p_pos_dn[i_batch, 0]] = 1.0



