import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image
import os
import numpy as np

#writer = SummaryWriter()
ADCN_aug_image0 = np.load('./ADCN_aug_images0.npy')
ADCN_image6 = np.load('./ADCN_images0.npy')
#ADCN_aug_image0[0:8].shape
#torch.Tensor(ADCN_aug_image0[0]).shape

root_dir = './'
tb_dir = os.path.join(root_dir, "tensorboard")
plot_2d_or_3d_image(data=torch.Tensor(ADCN_aug_image0[0:8]), step=0, writer=SummaryWriter(log_dir=tb_dir), frame_dim=-1)
plot_2d_or_3d_image(data=torch.Tensor(ADCN_image6[0:8]), step=1, writer=SummaryWriter(log_dir=tb_dir), frame_dim=-1)
