import math
import numpy as np
import torch
import argparse
from itertools import chain
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from model.base_model import Base_Model


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return hash.apply(input)


class DebiasNtXentLoss(nn.Module):
    def __init__(self, temperature, rho):
        super(DebiasNtXentLoss, self).__init__()
        self.temperature = temperature
        self.rho = rho
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.NLLLoss(reduction='sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j, device):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.exp(self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature)

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)
        N_neg = N - 2
        Ng = (-self.rho * N_neg * positive_samples + negative_samples.sum(-1, keepdim=True)) / (1 - self.rho)
        Ng = torch.clamp(Ng, min=N_neg * np.e ** (-1 / self.temperature))

        loss = (-torch.log(positive_samples / (positive_samples + Ng))).mean()
        return loss


def hswd(x, y):
    x_sorted = torch.sort(x, dim=0)[0]
    y_sorted = torch.sort(y, dim=0)[0]
    return torch.sqrt(((x_sorted - y_sorted) ** 2).sum(dim=0).mean())


class SCCH(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False        
        self.encoder = nn.Sequential(nn.Linear(4096, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, self.hparams.encode_length),
                                      )
        
        self.criterion = DebiasNtXentLoss(self.hparams.temperature, self.hparams.rho)
    
    def forward(self, imgi, imgj, alpha, device):
        imgi = self.vgg.features(imgi)
        imgi = imgi.view(imgi.size(0), -1)
        imgi = self.vgg.classifier(imgi)
        z_i = self.encoder(imgi)
        z_i = torch.tanh(F.normalize(z_i, 2, -1) * self.hparams.radius)

        imgj = self.vgg.features(imgj)
        imgj = imgj.view(imgj.size(0), -1)
        imgj = self.vgg.classifier(imgj)
        z_j = self.encoder(imgj)
        z_j = torch.tanh(F.normalize(z_j, 2, -1) * self.hparams.radius)

        contra_loss = self.criterion(z_i, z_j, device)
        batch_size = z_i.shape[0]

        p2 = torch.softmax(F.cosine_similarity(z_j[:, None, :], z_j[None, :, :], dim=2).fill_diagonal_(float('-inf')) / self.hparams.ts, dim=-1)
        w = 0.5 * torch.eye(batch_size).to(p2) + 0.5 * p2
        p1 = torch.softmax(F.cosine_similarity(z_i[:, None, :], z_j[None, :, :], dim=2) / self.hparams.temperature, dim=-1)
        sc_loss = (-w * torch.log(p1)).sum(1).mean()

        target = torch.cat((-torch.ones(batch_size // 2, self.hparams.encode_length), torch.ones(batch_size // 2, self.hparams.encode_length)), dim=0).to(z_i)
        q_loss = (hswd(z_i, target) + hswd(z_j, target)) / 2
        loss = contra_loss + alpha * self.hparams.sc * sc_loss + self.hparams.weight * q_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'sc_loss': sc_loss, 'q_loss': q_loss}
    
    def encode_discrete(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        z = torch.sign(self.encoder(x))
        return z

    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.encoder.parameters()}], lr = self.hparams.lr)

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument('-w',"--weight", default = 0.5, type=float,
                            help='quantization loss weight')
        parser.add_argument('-r',"--radius", default = 3, type=float)
        parser.add_argument('--rho', default = 0.05, type=float)
        parser.add_argument('--ts', default = 0.05, type=float)
        parser.add_argument('-s', '--sc', default=1, type=float)
        return parser