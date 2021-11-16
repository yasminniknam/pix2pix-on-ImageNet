import os
import torch
import numpy as np
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

def occlude(image):
    size = np.random.randint(30, 50)
    mask = np.zeros((size, size, 3))
    random = np.random.randint(0,200)
    new_mask = torch.from_numpy(mask)
    # print(image[0, :, random:100+random, random:100+random])
    image[random:size+random, random:size+random, :] = new_mask
    # print(image[0, :, random:100+random, random:100+random])
    return image

img_transform = transforms.Compose([transforms.ToTensor(),])

addr='./vall'

imageset = datasets.ImageFolder(root=addr,  transform=img_transform)

filenames= [imageset.imgs[i][0][11:-5] for i in range(len(imageset.imgs)) ]
print(filenames)
dataloader = torch.utils.data.DataLoader(imageset,batch_size=1,shuffle=False, num_workers=8)
save_path = '/home/yasamin/scratch/pix2pix/pix2pix-on-ImageNet/results/occluded_input'
#os.makedirs(save_path, exist_ok=True)
for idx, (img, y) in enumerate(dataloader):
    if idx == 0:
        print(y)
        print(idx)
        print(img.shape)
    new_img = occlude(img)
    print(filenames[idx], ' done!')
    save_image(new_img, os.path.join(save_path, filenames[idx] + '.jpeg'), nrow=8, normalize=True)
    # break

