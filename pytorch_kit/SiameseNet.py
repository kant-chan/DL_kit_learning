import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


'''
att_face dataset: https://files.cnblogs.com/files/king-lps/att_faces.zip
'''

class Config():
    root = './data/att_faces/'
    train_txt = os.path.join(root, 'train.txt')
    training_dir = './data/att_faces/train'
    testing_dir = './data/att_faces/test'
    epochs = 30
    batch_size = 4
Config = Config()


def convert():
    '''
    prepare data
    '''
    lines = []
    p_dirs = os.listdir(Config.root)
    for s in p_dirs:
        if s.startswith('s'):
            label = str(int(s[1:]) - 1)
            s = os.path.join(Config.root, s)
            lines_ = [os.path.join(s, i) + ' ' + label + '\n' for i in os.listdir(s)]
            lines += lines_
    # print(lines[:12])
    with open(Config.train_txt, 'w') as f:
        f.writelines(lines)
        print('train txt generate success.')

class NetDataset(Dataset):
    def __init__(self, image_folder_datasets, transform=None, should_invert=True):
        self.image_folder_datasets = image_folder_datasets
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.image_folder_datasets.imgs)
        # assure approx 0.5 images are in the same class
        get_same_cls = random.randint(0, 1)
        if get_same_cls:
            while True:
                img1_tuple = random.choice(self.image_folder_datasets.imgs)
                if img1_tuple[1] == img0_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.image_folder_datasets.imgs)
                if img1_tuple[1] != img0_tuple[1]:
                    break
        
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = torch.from_numpy(np.array(int(img0_tuple[1]!=img1_tuple[1]), dtype=np.float32))
        return img0, img1, label

    def __len__(self):
        return len(self.image_folder_datasets.imgs)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        euclidean_distance = F.pairwise_distance(out1, out2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=.0), 2))
        
        return loss


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        # 100
        self.cnn1 = nn.Sequential(nn.ReflectionPad2d(1),
                                  nn.Conv2d(1, 4, 3),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(4),
                                  nn.Dropout2d(p=.2),
                                  
                                  nn.ReflectionPad2d(1),
                                  nn.Conv2d(4, 8, 3),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(8),
                                  nn.Dropout2d(p=.2),
                                  
                                  nn.ReflectionPad2d(1),
                                  nn.Conv2d(8, 8, 3),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(8),
                                  nn.Dropout2d(p=.2))
        self.fc1 = nn.Sequential(nn.Linear(8 * 100 * 100, 500),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(500, 500),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(500, 5))
    
    def forward_once(self, x):
        out = self.cnn1(x)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataloader
folder_dataset = datasets.ImageFolder(root=Config.training_dir)
# print(folder_dataset.imgs)
siamese_dataset = NetDataset(image_folder_datasets=folder_dataset,
                             transform=transforms.Compose([transforms.Resize((100, 100)),
                                                           transforms.ToTensor()]),
                             should_invert=False)


net = SiameseNet().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
# print(net)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=Config.batch_size)


for epoch in range(0, Config.epochs):
    for i, data in enumerate(train_dataloader):
        img0, img1, label = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        
        optimizer.zero_grad()
        out0, out1 = net(img0, img1)
        loss = criterion(out0, out1, label)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('epoch {:3d}, loss {:4f}'.format(epoch, loss.item()))



####### utils ########
def imshow(img):
    npimg = img.numpy()  # tensor to numpy
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def vis():
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    # print(example_batch[0].size())  #torch.Size([8, 1, 100, 100])
    concat_img = torch.cat((example_batch[0], example_batch[1]), 0)
    # print(concat_img.size())  #torch.Size([16, 1, 100, 100])
    imshow(torchvision.utils.make_grid(concat_img))
    print(example_batch[2].numpy())


# if __name__ == '__main__':
#     # vis()
#     pass