import torch
import torch.nn as nn
from torch.nn import Sequential


class Flatten(nn.Module):
  """
  Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
  """
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return torch.squeeze(x)


def all_cnn_module():
  """
  Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
  https://arxiv.org/pdf/1412.6806.pdf
  Use a AvgPool2d to pool and then your Flatten layer as your final layers.
  You should have a total of exactly 23 layers of types:
  - nn.Dropout
  - nn.Conv2d
  - nn.ReLU
  - nn.AvgPool2d
  - Flatten
  :return: a nn.Sequential model
  """

  rl = nn.ReLU()
  l0 = nn.Dropout(p=0.2)
  l1 = nn.Conv2d(3, 96, 3, stride=1, padding=1)
  l3 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
  l5 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
  l7 = nn.Dropout(p=0.5)
  l8 = nn.Conv2d(96, 192, 3, stride=1, padding=1)
  l10 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
  l12 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
  l14 = nn.Dropout(p=0.5)
  l15 = nn.Conv2d(192, 192, 3, stride=1, padding=0)
  l17 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
  l19 = nn.Conv2d(192, 10, 1, stride=1, padding=0)
  l21 = nn.AvgPool2d(6)
  l22 = Flatten()

  return Sequential(l0,   l1, rl,  l3, rl,  l5, rl,
                    l7,   l8, rl, l10, rl, l12, rl,
                    l14, l15, rl, l17, rl, l19, rl,
                    l21, l22)
