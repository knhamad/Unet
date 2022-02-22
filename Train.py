from Model import UNetmodel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
cuda =  torch.cuda.is_available() and True

def train(epoch, Unet=UNetmodel.UNet,optimizer=optim,loader_train=data.DataLoader,criterion=nn):

    Unet.train()
    train_loss = torch.zeros(len(loader_train))
    for i, (img, label) in enumerate(loader_train):
        #import pdb; pdb.set_trace()
        if cuda:            
            img = img.cuda()
            label = label.cuda()
        #print(torch.unique(img))
        label=label/255
        output = Unet(img)
        label=label.reshape((output.shape[0], 1, 608, 968))
        label=label.float()     
        loss = criterion(output, label)
        
        train_loss[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print ("epoch %d step %d, loss=%.8f" %(epoch, i, loss.item()))
    if epoch % 10 == 0:    
        torch.save(Unet.state_dict(), 'ReanutTrial4UnetD6_%d.pth' % (epoch))
    #np.savetxt("D6train_loss%d" %(epoch+71), train_loss.numpy())
    return train_loss