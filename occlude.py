import os
import torch
import numpy as np
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

def occlude(image):
    mask = np.zeros((3, 100, 100))
    random = np.random.randint(0,100)
    new_mask = torch.from_numpy(mask)
    # print(image[0, :, random:100+random, random:100+random])
    image[0, :, random:100+random, random:100+random] = new_mask
    # print(image[0, :, random:100+random, random:100+random])
    return image

img_transform = transforms.Compose([transforms.ToTensor(),])

addr='./data/test'
data_addr = os.path.join(addr, '156imagesC')
imageset = datasets.ImageFolder(root=data_addr,  transform=img_transform)
filenames= [imageset.imgs[i][0][-7:-4] for i in range(len(imageset.imgs)) ]
dataloader = torch.utils.data.DataLoader(imageset,batch_size=1,shuffle=False, num_workers=8)
save_path = os.path.join(addr, 'image_occluded_reconstruction/occlude')
os.makedirs(save_path, exist_ok=True)
for idx, (img, y) in enumerate(dataloader):
    if idx == 0:
        print(y)
        print(idx)
        print(img.shape)
    new_img = occlude(img)
    save_image(new_img, os.path.join(save_path, filenames[idx] + '.png'), nrow=8, normalize=True)
    # break

