import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io
import torch.optim as optim
from model import DeepQSM
from loss import dipole_loss
import argparse


class TrainDataset(Dataset):
        def __init__(self, mat_files):
            self.training = True


        def __len__(self):
            return len(mat_files)

        def __getitem__(self, idx):
            file_name = mat_files[idx]           
            data = scipy.io.loadmat(file_name)
            phs  = torch.tensor(data['phs']).unsqueeze(dim=0)
            return phs     


def Train(model, start_epoch, n_epochs=10):

    for epoch in range(start_epoch, start_epoch+ n_epochs):
        model.train()
        total_loss = 0

        for _, data in enumerate(train_loader):
            data = data.cuda()
            data = (data - a) / x
            
            out = model(data)
            
            loss = dipole_loss(out,data)
            total_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
        print(f"Loss: {total_loss / batch_size}, epoch: {epoch+1}")

    save_checkpoint(model,optimizer,n_epochs, total_loss / batch_size)



def save_checkpoint(model, optimizer, epoch, loss, path="model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Susceptibility Mapping')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='num epochs')
    parser.add_argument('--resume', default=False, type=bool, help='chckpt resume')
    parser.add_argument('--bs', default=10, type=int, help='Batch Size')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    n_epochs = args.epochs
    batch_size = args.bs # batch size
    resume = args.resume

    mat_files = [f'training_data/tr-1-1-{i}.mat' for i in range(120, 171)]


    train_dataset = TrainDataset(mat_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Data distribution - mean and variance
    data_dist = sio.loadmat('training_data/tr-stats.mat')
    a = torch.tensor(data_dist['inp_mean']).cuda()
    b  = torch.tensor(data_dist['out_mean']).cuda()
    x  = torch.tensor(data_dist['inp_std' ]).cuda()
    y  = torch.tensor(data_dist['out_std' ]).cuda() 


    model = DeepQSM().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_epoch = 0
    if resume:
    # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load("model_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']


    Train(model, start_epoch, n_epochs=n_epochs)

    
        