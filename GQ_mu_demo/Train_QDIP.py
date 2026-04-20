import os
import shutil
import time
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.io import loadmat, savemat

from QDIP.QDIP import QDIP

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


torch.manual_seed(1029)
np.random.seed(10)
gen = torch.Generator()
gen.manual_seed(0)


def tensor_to_matrix(x):
    x = rearrange(x, 'c h w -> c (h w)')
    return x


def matrix_to_tensor(x, shape):
    x = rearrange(x, 'c (h w) -> c h w', h=shape[0], w=shape[1])
    return x


def init_optimizer(all_parameter):
    opt = torch.optim.Adam(all_parameter, lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    return opt, scheduler


def init_data():

    pth = "QNN/QAT/initial.mat"
    data_mat = loadmat(pth)

    A = data_mat['A_0']
    epoch = data_mat['epoch']
    HSI = data_mat['Zh_tensor']

    HSI = torch.from_numpy(HSI).permute(2, 0, 1).type(torch.FloatTensor)
    H = HSI.shape[1]
    W = HSI.shape[2]
    HSI = rearrange(HSI, 'c h w -> c (h w)')

    A = torch.from_numpy(A).type(torch.FloatTensor)
    N = A.shape[-1]

    all_dict = {
        "N": N,
        "L": H * W,
        "H": H,
        "W": W,
        "A": A,
        "epoch": epoch.item(),
        "HSI": HSI,
    }
    return all_dict



class Train(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = "./QNN/QAT/"
        self.pretrain_ind = True
        self.loss = nn.MSELoss()

    def optimization(self, train_data):
        train_loss_list_G = []
        Zh = train_data['HSI'].to(self.train_dev)
        A = train_data['A'].to(self.train_dev)

        with torch.autograd.set_detect_anomaly(True):
            self.net.train()
            self.optG.zero_grad()
            S_QNN = self.net()
            errG = self.loss(A @ S_QNN, Zh)
            errG.backward()
            self.optG.step()
            self.scheduler.step()

        train_loss_list_G.append(errG.item())
        return S_QNN, errG.item()

    def forward(self):
        self.train_data = init_data()
        self.H = self.train_data['H']
        self.W = self.train_data['W']
        self.epoch_num = self.train_data['epoch']

        self.net = QDIP(
            device=self.train_dev,
            out_channel=6,
        ).to(self.train_dev)
        self.optG, self.scheduler = init_optimizer(self.net.parameters())

        t1 = time.time()


        for i in range(self.epoch_num):
            S_QNN, train_loss_G = self.optimization(self.train_data)
            msg = "[Epoch_{}] QNN Loss: {} ".format(i, train_loss_G)
            print(msg)

        S_QNN = S_QNN.detach().cpu().numpy()
        S_QNN = rearrange(
            S_QNN,
            'c (h w) -> c h w',
            h=self.train_data["H"],
            w=self.train_data["W"],
        )

        t2 = time.time()
        time_QDIP = t2 - t1
        all_dict = {
            "S_QNN": S_QNN,
            "Times": time_QDIP,
        }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        savemat(os.path.join(self.save_path, 'QNN.mat'), all_dict)


        print('=================== Training is successfully done ========================')
        print('time:', t2 - t1)


if __name__ == "__main__":
    model = Train()
    model()