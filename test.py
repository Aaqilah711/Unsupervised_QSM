import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io
from model import DeepQSM
import argparse
from  metrics import compute_psnr, compute_ssim

class TestDataset(Dataset):
    def __init__(self, mat_files):
        self.training = True
        self.mat_files = mat_files


    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        file_name = self.mat_files[idx]           
        data = scipy.io.loadmat(file_name)
        phs  = torch.tensor(data['phs']).unsqueeze(dim=0)
        susc = torch.tensor(data['susc']).unsqueeze(dim=0)
        mask = torch.tensor(data['msk']).unsqueeze(dim=0)
        return phs,susc, mask     


def Test(model):
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for phs,susc,mask in (test_dataloader):
            phs = phs.cuda()
            phs = (phs - a) / x
            out = model(phs)
            out = (out).detach().cpu()

            susc = (susc.cuda() - b) / y
            masked_susc = (susc.detach().cpu() * mask)
            
            masked_out = (out * mask)
            psnr_data = compute_psnr(masked_out, masked_susc)
            mssim, _ = compute_ssim(masked_out, masked_susc)
            total_psnr += psnr_data
            total_ssim += mssim

    psnr_avg = total_psnr / len(test_dataloader)
    ssim_avg = total_ssim / len(test_dataloader)

    print(f"PSNR: {psnr_avg}, SSIM {ssim_avg.item()}")


if __name__ == "__main__":

    test_mat_files = [f'testing_data/tr-2-4-{i}.mat' for i in range(63, 102)]

    test_dataset = TestDataset(test_mat_files)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Data distribution - mean and variance
    data_dist = sio.loadmat('training_data/tr-stats.mat')
    a = torch.tensor(data_dist['inp_mean']).cuda()
    b  = torch.tensor(data_dist['out_mean']).cuda()
    x  = torch.tensor(data_dist['inp_std' ]).cuda()
    y  = torch.tensor(data_dist['out_std' ]).cuda() 

    model = DeepQSM().cuda()


    print('==> Loading checkpoint..')
    checkpoint = torch.load("model_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    Test(model)