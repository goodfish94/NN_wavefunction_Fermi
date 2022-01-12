import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
def check_consistent(x, R):
    """
    check consistent btw x,  R
    :param x:
    :param R:
    :return:
    """
    def if_consistent( x_, R_):



        R_ = R_.long().cpu().numpy()
        R_set = set(R_)
        for i in range(0,len(x_) ):
            if( i in R_set  ):
                if( x_[i,1] > 0.5 and x_[i,0] < 0.5 ):
                    continue
                else:
                    print(" x inconsistnet")
                    return False
            else:
                if (x_[i, 1] < 0.5 and x_[i, 0] > 0.5):
                    continue
                else:
                    print(" x inconsistnet")
                    return False

        return True

    batch, n_orb,_ = R.shape
    for i in range(batch):
        for j in range(n_orb):
            if( if_consistent(x[i,j], R[i,j]) ):
                continue
            else:
                print("x ")
                print(x[i,j])
                print("R")
                print(R[i,j])
                print("R one hot")
                return False

    return True
def generate_R_R_onehot(x,N):
    """
    generate R,R_onehot from x
    :param x: (*, norb, L, 2)
    :return:
    """

    batch, norb, L, _ = x.shape
    R = torch.zeros(batch, norb, N)
    for i in range(0,batch):
        for j in range(0,norb):
            it = 0
            for k in range(0,L):
                if(x[i,j,k,1] > 0.5):
                    R[i,j,it] = k
                    it+= 1
            if( it != N):
                raise ValueError("x and fill N don't match")

    R_onehot = F.one_hot(R.long(), num_classes=L)
    R_onehot = torch.transpose(R_onehot, dim0=2, dim1=3)
    return R, R_onehot

def random_init(nbatch ,norbit = 2, L = 9,N=4, device = 'cuda'):
    x = torch.zeros( nbatch,norbit, L,2 )
    R = torch.zeros(nbatch, norbit, N)

    # assume N<L/2
    for i in range(nbatch):

        for i_orb in range(norbit):
            perm = torch.randperm(L)
            for j in range(0,L): # set all unfilled
                x[i,i_orb,j,0] = 1.0
            for j in range(0,N):
                r = perm[j]
                x[i, i_orb, r, 1] = 1.0
                x[i, i_orb, r, 0] = 0.0
                R[i,i_orb,j] = r
    R=R.long()

    return x, R



def non_int_slater_mat_2d(Lx,Ly,N,t):
    """
    return slater matrix of shape L,N tensor, corresponds to the dispersion 2t(cos(kx)+sin(ky))
    :param L:
    :param N:
    :return:
    """

    kx = np.asarray(range(0, Lx )) * 2.0 * np.pi / Lx # half kx is enough
    ky = np.asarray(range(0, int(Ly))) * 2.0 * np.pi / Ly
    rx = np.asarray(range(0,int(Lx)))
    ry = np.asarray(range(0,int(Ly)))
    rx,ry = np.meshgrid(rx,ry)
    ep = []
    for x in kx:
        for y in ky:
            ep.append( (2.0*t*(np.cos(x) + np.cos(y) ), x,y) )
    ep = sorted(ep, key=lambda x:x[0])

    pltx = []
    plty = []
    sm_complex = np.zeros((Lx * Ly, N), dtype=np.complex)
    for i in range(0, N):
        en, kx, ky = ep[i]
        sm_complex[:, i] = np.reshape(np.exp(1j * (kx * rx + ky * ry)), -1)
        pltx.append(kx * Lx / 2.0 / np.pi - 1)
        plty.append(ky * Ly / 2.0 / np.pi - 1)
    # plt.plot(pltx, plty, '<')
    # plt.show()


    kx = np.asarray(range(0, int(Lx/2) +1 )) * 2.0 * np.pi / Lx # half kx is enough
    ky = np.asarray(range(0, int(Ly))) * 2.0 * np.pi / Ly
    rx = np.asarray(range(0,int(Lx)))
    ry = np.asarray(range(0,int(Ly)))
    rx,ry = np.meshgrid(rx,ry)

    sm = np.zeros((Lx*Ly,N))

    ep = []
    for x in kx:
        for y in ky:
            ep.append( (2.0*t*(np.cos(x) + np.cos(y) ), x,y) )
    ep = sorted(ep, key=lambda x:x[0])


    it= 0
    pltx = []
    plty = []

    for i in range(0,N):
        en,kx,ky = ep[i]

        if( ( np.abs(kx - 0)<0.001 or np.abs(kx-np.pi)<0.001) and (np.abs(ky - 0)<0.001 or np.abs(ky-np.pi)<0.001) ):
            sm[:,it] = np.reshape(np.cos(kx*rx+ky*ry), -1)

            it += 1
            pltx.append(kx*Lx/2.0/np.pi-1.0)
            plty.append(ky*Ly/2.0/np.pi-1.0)
            # print("type 1", kx/2.0/np.pi * Lx, ky/2.0/np.pi * Ly,en)
            if( it == N):
                break


        else:
            if( np.abs(kx-np.pi) < 0.001 and ky < np.pi ):
                continue

            x,y = kx/2.0/np.pi * Lx, ky/2.0/np.pi * Ly
            # print("type 2", x,y, en)

            pltx.append(kx*Lx/2.0/np.pi-1)
            plty.append(ky*Ly/2.0/np.pi-1)
            sm[:,it] = np.reshape( np.cos(kx*rx + ky*ry), -1)

            it += 1
            if (it == N):
                break
            sm[:,it] = np.reshape( np.sin(kx*rx + ky*ry),-1 )

            pltx.append((2.0*np.pi - kx)*Lx/2.0/np.pi-1)
            plty.append((2.0*np.pi - ky)*Ly/2.0/np.pi-1)
            x, y = (2.0*np.pi - kx) / 2.0 / np.pi * Lx, (2.0*np.pi - ky) / 2.0 / np.pi * Ly
            # print("type 2", x, y, en)
            it += 1
            if (it == N):
                break
    sm = sm # /np.sqrt(Lx*Ly)
    # plt.plot(pltx, plty ,'o')
    # plt.show()
    return sm



def get_non_int_energy( L, N, t):
    """
    one orbital, square lattice, N=fill, L=size, t= hopping
    :param L:
    :param N:
    :param t:
    :return:
    """
    kx = np.asarray( range(0,L) ) * 2.0 * np.pi/L
    ky = np.asarray( range(0,L) ) * 2.0 * np.pi/L

    ep = []
    for x in kx:
        for y in ky:
            ep.append( 2.0*t*(np.cos(x) + np.cos(y) ) )

    ep = np.asarray( sorted(ep) )

    return np.sum(ep[0:N])




def get_non_int_energy_1d( L, N, t):
    """
    one orbital, N=fill, L=size, t= hopping
    :param L:
    :param N:
    :param t:
    :return:
    """
    kx = np.asarray( range(0,L) ) * 2.0 * np.pi/L

    ep = []
    for x in kx:
        ep.append( 2.0*t*(np.cos(x) ) )

    ep = np.asarray( sorted(ep) )

    return np.sum(ep[0:N])


def get_corr_non_int(L,N,t):
    """
    get charge and mag correlation for non interatin sys
    :param L:
    :param N:
    :param t:
    :return:
    """
    kx = np.asarray(range(0, L)) * 2.0 * np.pi / L
    ky = np.asarray(range(0, L)) * 2.0 * np.pi / L


    g = np.zeros((L,L), dtype=np.complex) # cdag_(i+r) cdagc_i
    ep = []
    for x in kx:
        for y in ky:
           ep.append( [2.0*t*(np.cos(x) + np.cos(y) ), x,y] )
    ep = np.asarray( sorted(ep) )[0:N]

    kx, ky, ek = ep[:,1], ep[:,2], ep[:,0]
    for i in range(L):
        for j in range(L):
            g[i,j] = np.sum( np.exp(-1j * (kx*i + ky*j) ) )/L/L

    n_equal = (N/L/L) **2 + g * np.conj(g) # equal spin
    n_anti  = (N/L/L) **2

    nn = 2.0 * n_equal + 2.0 * n_anti
    mzmz = 2.0 * n_equal - 2.0 * n_anti

    data = {}
    data['nn'] = nn
    data['mzmz'] = mzmz

    return data

if __name__ == '__main__':
    Lx = 6
    Ly = Lx
    N = int(Lx*Ly/2)
    t= 1.0

    #
    sm = non_int_slater_mat_2d(Lx,Ly,N,t)
    # # print(sm)
    #
    # ind_ = np.random.choice(Lx*Ly,size = N , replace = False )
    # # ind_ = [3,4,5,0]
    for i in range(0,10):
        ind_ = np.random.choice(Lx * Ly, size=N, replace=False)
        det1 = np.linalg.det(sm[ind_,:])
        print(np.log(np.abs(det1)))
    # print(sm[ind_,:])
    #
    # rx = np.asarray(range(0, int(Lx)))
    # ry = np.asarray(range(0, int(Ly)))
    # rx, ry = np.meshgrid(rx, ry)
    # rx = np.reshape(rx,-1)
    # ry = np.reshape(ry,-1)
    # for i in ind_:
    #     print("r = ", rx[i],ry[i])
    # # fig, ax = plt.subplots(ncols=2)
    # # for i in range(N):
    # #     plt.plot(sm[:,i],'-o')
    # # plt.show()