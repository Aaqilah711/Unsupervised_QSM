import numpy as np
import torch
import torch.nn as nn

def dipole_kernel(matrix_size, voxel_size, B0_dir=[0,0,1]):
    [Y,X,Z] = np.meshgrid(np.linspace(-np.int(matrix_size[1]/2),np.int(matrix_size[1]/2)-1, matrix_size[1]),
                       np.linspace(-np.int(matrix_size[0]/2),np.int(matrix_size[0]/2)-1, matrix_size[0]),
                       np.linspace(-np.int(matrix_size[2]/2),np.int(matrix_size[2]/2)-1, matrix_size[2]))
    X = X/(matrix_size[0])*voxel_size[0]
    Y = Y/(matrix_size[1])*voxel_size[1]
    Z = Z/(matrix_size[2])*voxel_size[2]
    D = 1/3 - np.divide(np.square(X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2]), np.square(X)+np.square(Y)+np.square(Z) + np.finfo(float).eps )
    D = np.where(np.isnan(D),0,D)

    D = np.roll(D,np.int(np.floor(matrix_size[0]/2)),axis=0)
    D = np.roll(D,np.int(np.floor(matrix_size[1]/2)),axis=1)
    D = np.roll(D,np.int(np.floor(matrix_size[2]/2)),axis=2)
    D = np.float32(D)
    D = torch.tensor(D).unsqueeze(dim=0)
    
    return D


def dipole_loss(out,data):
    phi =  data
    susc = out
    f_susc = torch.fft.fftn(susc,dim=[2,3,4])
    matrix_size = [64, 64, 64]
    voxel_size = [1,  1,  1]

    dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
    dk=dk.float()
    dk = torch.unsqueeze(dk, dim=0).cuda()
    
    f_res = dk * f_susc
    phi_cap = torch.real(torch.fft.ifftn(f_res, dim=[2,3,4]))

    mse_loss = nn.MSELoss()
    loss = mse_loss(phi, phi_cap)
    return loss


def compute_local_field(susc):
    f_susc = torch.fft.fftn(susc,dim=[2,3,4])
    matrix_size = [64, 64, 64]
    voxel_size = [1,  1,  1]

    dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
    dk=dk.float()
    dk = torch.unsqueeze(dk, dim=0).cuda()
    
    f_res = dk * f_susc
    phi = torch.real(torch.fft.ifftn(f_res, dim=[2,3,4]))

    return phi