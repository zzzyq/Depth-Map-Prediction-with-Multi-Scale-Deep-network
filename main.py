import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import torch.optim as optim
import time
from tqdm import tqdm

from model import *
from IO import *

dataset = data(root_dir='./',transform=transforms.Compose([
                                                   Rescale((input_height, input_width),(output_height, output_width)),
                                                   ToTensor()]))
dataset_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
net = coarseNet()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

def train(model, criterion, optimizer,n_epochs,print_every):
    start = time.time()
    losses = []
    print("Training for %d epochs..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = 0

        for data in dataset_loader:
            # get the inputs
            inputs=data["image"]
            depths=data["depth"]
        
            # wrap them in Variable
            inputs, depths = Variable(inputs), Variable(depths)

            # zero the parameter gradients
            optimizer.zero_grad()
                
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss += loss.data[0] 
            loss.backward()
            optimizer.step()
        
        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            print(loss, '\n')

    return losses

loss = train(net,criterion,optimizer,5,1)