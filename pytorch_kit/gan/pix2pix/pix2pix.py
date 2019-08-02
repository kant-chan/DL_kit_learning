import os, sys, math
import time, datetime
import itertools
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import ImageDataset
from model import *

# hyper parameters
epoch = 0
n_epochs = 200
dataset_name = 'facades'
batch_size = 1
lr = 2e-4
b1 = 0.5
b2 = 0.999
decay_epoch = 100
n_cpu = 8
img_height = 256
img_width = 256
channels = 3
sample_interval = 500
checkpoint_interval = -1

os.makedirs('images/{}'.format(dataset_name), exist_ok=True)
os.makedirs('saved_models/{}'.format(dataset_name), exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loss function
criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()
lambda_pixel = 100

# calculate output of image discriminator (PatchGAN)
patch = (1, img_height // 16, img_width // 16)

generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)

# init param
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimize
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))



transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = torch.utils.data.DataLoader(
    ImageDataset('../../data/{}'.format(dataset_name), transforms_=transforms_),
    batch_size=batch_size
    shuffle=True
    num_workers=n_cpu
)
val_dataloader = torch.utils.data.DataLoader(
    ImageDataset('../../data/{}'.format(dataset_name), transforms_=transforms_, mode='val'),
    batch_size=10,
    shuffle=True,
    num_workers=1
)

for epch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):
        real_a, real_b = batch['img_b'], batch['img_a']
        real_a, real_b = real_a.to(device), real_b.to(device)

        valid = torch.ones(real_a.size(0), *patch, requires_grad=True)
        fake = torch.zeros(real_b.size(0), *patch, requires_grad=True)

        ##### train generator
        optimizer_G.zero_grad()
        fake_b = generator(real_a)
        pred_fake = discriminator(fake_b, real_a)
        loss_GAN = criterion_GAN(fake_b, real_a)
        loss_pixel = criterion_pixelwise(fake_b, real_b)