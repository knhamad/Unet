from Model import UNetmodel
import torch
from torch.utils import data
import torch.nn as nn
cuda =  torch.cuda.is_available() and True

def test(Unet=UNetmodel.UNet,loader_test=data.DataLoader,criterion=nn):
    
    Unet.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(loader_test):
    
            if cuda:
                img = img.cuda()
                label = label.cuda()
        
            output = Unet(img)
            label=label/255
            label=label.reshape((output.shape[0], 1, 608, 968))
            label=label.float()
            test_loss += criterion(output, label).item()
       
    test_loss /= len(loader_test.dataset)
    print (" loss=%.8f" %(test_loss))

    return test_loss