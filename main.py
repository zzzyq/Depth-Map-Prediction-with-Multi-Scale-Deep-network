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
coarse_net = coarseNet()
optimizer = optim.SGD(coarse_net.parameters(), lr=0.01)
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
            loss = criterion(outputs, depths)
            loss += loss.data[0] 
            loss.backward()
            optimizer.step()
        
        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            print(loss, '\n')

    return losses


def show_img(sample):
    images, depth = sample['image'], sample['depth']
    img = Variable(images)
    output = coarse_net(img)
    
    fig=plt.figure()
    fig.add_subplot(1,3,1)
    plt.imshow(img.data[0].numpy().transpose((1, 2, 0)))
    fig.add_subplot(1,3,2)
    plt.imshow(Variable(depth).data[0].numpy())
    fig.add_subplot(1,3,3)
    plt.imshow(output.data[0].numpy())
    plt.show()

loss = train(coarse_net,criterion,optimizer,5,1)
show_img(iter(dataset_loader).next())