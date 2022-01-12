# Traning function

import torch
from slater_wf import slater_wf
from mcmc import Sampler
import time


def train(config, wf, sampler, optimizer, x, R, save_path="ckpt"):
    warmup = config['warmup']
    i_eval = config['i_eval']
    i_update = config['i_update']
    i_iter = config['i_iter']

    Lx, Ly = config['Lx'], config['Ly']
    N = config['N']
    device = config['device']

    with torch.no_grad():
        log_abs_det = wf.get_log_abs_det(x, R)

    for i in range(0, warmup):
        with torch.no_grad():
            x, R, log_abs_det  = sampler.one_step(x, R, log_abs_det , wf)

    average_en = 0.0
    average_loss = 0.0
    it_update = 1
    sum_logdet = 0.0
    loss = 0.0

    wf.train()

    t_st = time.time()

    for it in range(i_iter * i_update):
        log_abs_det = wf.get_log_abs_det(x,R)
        for i in range(0, i_eval):
            with torch.no_grad():
                x,  R, log_abs_det  = sampler.one_step(x, R, log_abs_det , wf)

        SM = wf.get_SM(x, R)
        mean = torch.mean(torch.abs(SM), dim=(-1, -2), keepdim=True)
        logdet = torch.det(SM/mean)
        signdet = torch.sign(logdet).prod(dim=-1)
        logdet = torch.sum(torch.log(torch.abs(logdet)), dim=(-1)) + N * torch.sum((torch.log(mean)), dim=(1, 2, 3))
        with torch.no_grad():
            hx = wf.get_local_hx(x, R, logdet, signdet)



        # with torch.no_grad():
        # inv = torch.inverse(SM/mean)  # shape = (batch, 2, N,N )
        # inv = inv/mean
        # trace = torch.diagonal(torch.matmul(inv,SM), dim1=-1, dim2=-2).sum(dim=(-1,-2))


        en = torch.mean(hx[0])

        average_en = (average_en * (it_update - 1) + en) / (it_update)

        loss = torch.mean(hx[0] * logdet ) - average_en * torch.mean(logdet )
        loss.backward()

        if (it_update % i_update == 0):
            # loss += - ( sum_logdet * average_en )
            # loss = loss/i_update

            # loss.backward()
            print(it, i_update, it / i_update)
            # print(logphi)
            # print(loss)
            print("it = ", int(it / i_update), "En = ", average_en, en, " loss = ", average_loss, " time = ",
                  time.time() - t_st, " time per iter = ", (time.time() - t_st) / ((it / i_update)))

            optimizer.step()
            wf.zero_grad()
            optimizer.zero_grad()

            it_update = 0

            average_en = 0.0
            average_loss = 0.0
            sum_logdet = 0.0
            loss = 0.0

        if (it % (i_update * 10) == 0 and it != 0):
            torch.save(wf.nn_wf.state_dict(), save_path)
            print("save")
        it_update += 1


def get_local_meas(config, x):
    """

    :param x: shape = batch, norb,  L, nloc, fock space config [1,0] = empty, [0,1] = full
    :return: local obs
    """
    data = {}

    data['n_orb'] = torch.mean( x[:,:,:,1] ,dim=[0,2]) # shape = norb, mean fill of each orb
    data['double'] = torch.mean( x[:,0,:,1] * x[:,1,:,1], dim = [0,1]) # shape = (1,), mean Double occup
    # data['mz'] = torch.mean( x[:,0,:,1] - x[:,1,:,1], dim= [0,1]) # shape = (1,), mean mag
    data['mz'] = torch.mean(x[:, 0, :, 1] - x[:, 1, :, 1], dim=[0])  # shape = (L, ), mag

    return data

def get_density_correlation(config, x):
    """

    :param x: shape = batch, norb,  L, nloc, fock space config [1,0] = empty, [0,1] = full
    :return: density correlation
    """
    data = {}
    Lx,Ly = config['Lx'], config['Ly']
    device = config['device']

    data['nn'] = torch.zeros( (Lx,Ly) ).to(device)
    data['mzmz'] = torch.zeros( (Lx,Ly) ).to(device)

    fill = x[:,:,:,1] # filling number

    for rx in range(0,Lx):
        for ry in range(0,Ly):
            r1 = rx * Ly +ry
            n = fill[:,0,r1] + fill[:,1,r1] # filling
            mz = fill[:,0,r1] - fill[:,1,r1] # mz
            for i in range(0,Lx):
                ix = ( rx + i )%Lx
                for j in range(0,Ly):
                    iy = (ry+j) % Ly
                    r2 = ix * Ly + iy

                    n2 = fill[:,0,r2] + fill[:,1,r2]
                    mz2 = fill[:,0,r2] - fill[:,1,r2]

                    data['nn'][i,j] += torch.mean( n*n2 )
                    data['mzmz'][i,j] += torch.mean( mz*mz2 )

    data['nn'] /= (Lx*Ly)
    data['mzmz'] /= (Lx * Ly)

    return data

def evaluation(config, wf, sampler, x, R_onehot, R):

    warmup = config['warmup']
    i_eval = config['i_eval'] # one measurement every i_eval samples
    i_meas = config['i_meas'] # total measurement
    Lx, Ly = config['Lx'], config['Ly']
    device = config['device']

    with torch.no_grad():
        logdet,signdet = wf.get_logdet(x,R_onehot)

    for i in range(0,warmup):
        with torch.no_grad():
            x, R_onehot, R, logdet, signdet = sampler.one_step(x,R_onehot,R,logdet,signdet, wf)

    avg_en  = 0.0
    avg_mz  = 0.0 # nup -ndn
    avg_n_orb = 0.0
    avg_D = 0.0 # (nup * ndn)
    avg_nn = torch.zeros( (Lx,Ly) ).to(device) #1/Lx/Ly sum_r nup_r n_dn_(r+i)
    avg_mzmz = torch.zeros( (Lx,Ly) ).to(device) #1/Lx/Ly sum_r nup_r n_up_(r+i)

    wf.eval()

    t_st = time.time()
    for it in range(i_meas):

        for i in range(0, i_eval):
            with torch.no_grad():
                x, R_onehot, R, logdet,signdet = sampler.one_step(x, R_onehot, R, logdet, signdet, wf)

        with torch.no_grad():
            logdet, signdet, hx = wf(x, R_onehot,R) #

        en = torch.mean(hx[0])

        avg_en = (avg_en * (it) + en) /(it + 1 )

        data_loc = get_local_meas(config, x)
        data_corr = get_density_correlation(config,x)

        avg_mz = (avg_mz * it + data_loc['mz'] )/(it+1)
        avg_n_orb = (avg_n_orb * it + data_loc['n_orb'] )/(it+1)
        avg_D = (avg_D * it + data_loc['double']) / (it + 1)
        avg_nn = (avg_nn * it + (data_corr['nn'])) / (it+1)
        avg_mzmz = (avg_mzmz * it + (data_corr['mzmz'])) / (it+1)

        if( it%100 == 0 ):
            print(" it ", it , it/float(i_meas)," time = ", time.time()-t_st, " time per iter = ", (time.time()-t_st)/((it+1)))
            print("en = ", avg_en , en)

    data = {}
    data['En'] = avg_en
    data['n_orb'] = avg_n_orb
    data['D'] = avg_D
    data['mz'] = avg_mz
    data['nn'] = avg_nn
    data['mzmz'] = avg_mzmz

    for key in data:
        data[key] = data[key].cpu().numpy()

    return data