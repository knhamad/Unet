import os
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import glob
class DataSetLoader(Dataset):
    
    def __init__(self, root, train = 'True', mode = 'RGB', img_transform = None, label_transform = None, f=None):     
        
        self.root = root
        self.mode = mode
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []
        
        imgset_dir = os.path.join(self.root, r'C:\Users\k.hamad\Desktop\UF\PhD\Github\Pytorch-UNet\data')
        
        imgset_dir= os.path.join(imgset_dir, f)
        images=glob.glob(os.path.join(r'C:\Users\k.hamad\Desktop\UF\PhD\Github\Pytorch-UNet\data', f ,r'*.png'))
        for im in images:
            self.files.append({
                    "img": im,
                    "label": os.path.join(r'C:\Users\k.hamad\Desktop\UF\PhD\Github\Pytorch-UNet\data',f,r'labels',im[im.rfind('\\')+1:-4]+r'.npy')
                    })
    def __len__(self):     
        # print("files size in dataset class", len(self.files))
        return len(self.files)
    
    def __getitem__(self, index):
        
        datafiles = self.files[index]
        #print(datafiles)
        img_file = datafiles["img"]
        #print('hi')
        #print(img_file)
        #print('bye')
        if self.mode == 'RGB':
            img = Image.open(img_file).convert('RGB')
        if self.mode == 'gray':
            img = Image.open(img_file).convert('L')
            img = img.convert('RGB')
                    
        label_file = datafiles["label"]
        #print(label_file)
        # label = Image.open(label_file).convert("P")
        
        
        if self.img_transform is not None:
            img = self.img_transform(img)
            
         
       
        # label = np.expand_dims(np.array(label),2)*255
        #print(label_file)
        label_file = np.load(label_file)*255
        label_file = label_file.astype(int)
        
        if self.label_transform is not None:
            label_file = self.label_transform(label_file) 

        return img, label_file