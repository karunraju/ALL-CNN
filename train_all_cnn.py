"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from all_cnn import all_cnn_module
from preprocessing import cifar_10_preprocess


#DEBUG = True
DEBUG = False
BATCH_SIZE = 500
M = 0.9
LR = 0.01 
L2R = 0.001 
lverbose = 25 
STAT_EPOCH_MODEL = 3
save_state = True

if DEBUG:
  EPOCHS = 3
  save_state = False
else:
  EPOCHS = 10 
PATH = './'

class CifarTrainDataset(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.len = X.shape[0]  

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    cX = self.X[index, :, :, :]
    cY = np.array([self.Y[index]])
    cXt = torch.from_numpy(cX).double()
    cYt = torch.from_numpy(cY).long()
    return cXt, cYt


class CifarTestDataset(Dataset):
  def __init__(self, X):
    self.X = X
    self.len = X.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    cX = self.X[index, :, :, :]
    cXt = torch.from_numpy(cX).double()
    return cXt


def write_results(predictions, output_file='predictions.txt'):
  """
  Write predictions to file for submission.
  File should be:
    named 'predictions.txt'
    in the root of your tar file
  :param predictions: iterable of integers
  :param output_file:  path to output file.
  :return: None
  """
  with open(output_file, 'w') as f:
    for y in predictions:
      f.write("{}\n".format(y))


def test_model(net, data_loader, epoch, gpu):
  fname = PATH + 'predictions_' + str(epoch + 1) + '.txt'
  with open(fname, 'w') as f:
    with torch.no_grad():
      for tXt in data_loader:
        if gpu:
           tXt = tXt.cuda()
        tO = net(tXt)
        _, pred = torch.max(tO, dim=1)
        if gpu:
          pred = pred.cpu()
        pred = pred.numpy()
        for i in range(len(pred)):
          f.write("{}\n".format(pred[i]))


def test_dataset(net, criterion, data_loader, epoch, gpu, tag='Training'):
  accuracy = 0
  total_size = 0
  loss = 0
  itrs = 0
  with torch.no_grad():
    for iXt, iYt in data_loader:
      iYt = torch.squeeze(iYt)
      if gpu:
        iXt, iYt = iXt.cuda(), iYt.cuda()
      iO = net(iXt)
      loss_ = criterion(iO, iYt)
      _, pred = torch.max(iO, dim=1)
      if gpu:
        pred = pred.cpu()
        iYt = iYt.cpu()
      iY = iYt.numpy()
      accuracy =  accuracy + np.sum(np.isclose(pred.numpy(), iY.astype(int)))
      loss = loss + loss_.item()
      total_size = total_size + iY.shape[0]
      itrs = itrs + 1

    loss = loss/itrs
    print('Total Size:%d' % total_size)
    accuracy = accuracy/total_size
    print('[%d] %s loss: %.3f Accuracy: %.4f' % (epoch + 1, tag, loss, accuracy))


def main(filepath=None, test=False):
  path = './dataset/'
  train_data = np.load(path + 'train_feats.npy')
  train_labels = np.load(path + 'train_labels.npy')
  test_data = np.load(path + 'test_feats.npy')

  print('Preprocessing...')
  xtrain, xtest = cifar_10_preprocess(train_data, test_data)

  train_dataset = CifarTrainDataset(xtrain, train_labels)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
  train_loader_test = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
  ts_dataset = CifarTestDataset(xtest)
  test_loader = DataLoader(ts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

  net = all_cnn_module() 
  net = net.double()

  # Xavier Weight Initialization
  if filepath is None:
    for mod in net.modules():
      if isinstance(mod, (nn.Conv2d, nn.Linear)):
        print(type(mod))
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0)

  criterion = nn.CrossEntropyLoss()

  optimizer_SGD = optim.SGD(net.parameters(), lr=LR, momentum=M, weight_decay=L2R)

  # GPU availability
  gpu = torch.cuda.is_available()
  if gpu:
    print("Using GPU")
    net = net.cuda()
  else:
    print("Using CPU")

  # Resume or Test if file path is given
  if filepath is not None:
    state = torch.load(filepath)
    net.load_state_dict(state['state_dict'])
    optimizer_SGD.load_state_dict(state['optimizer'])
    if test:
      test_model(net, test_loader, 0, gpu)
      return

  print('Training...')
  dump_epoch = 0
  for epoch in range(EPOCHS):
    running_loss = 0.0
    i = 0
    for bXt, bYt in train_loader:
      bYt = torch.squeeze(bYt)
      if gpu:
        bXt, bYt = bXt.cuda(), bYt.cuda()

      # zero the gradients
      optimizer_SGD.zero_grad()

      # forward + backward + optimize
      bO = net(bXt)
      loss = criterion(bO, bYt)
      loss.backward()
      optimizer_SGD.step()

      # print statistics
      running_loss += loss.item()
      if i % lverbose == lverbose - 1:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / lverbose))
        running_loss = 0.0
      i = i + 1

    if epoch % STAT_EPOCH_MODEL == 0:
      test_model(net, test_loader, epoch, gpu)
      dump_epoch = epoch 
      print('Dumped Submission file at the epoch: %d' % (epoch + 1))
      test_dataset(net, criterion, train_loader_test, epoch, gpu, 'Training')

  if dump_epoch != epoch:
    # Dump submission file at the end
    test_model(net, test_loader, epoch, gpu)
    test_dataset(net, criterion, train_loader_test, epoch, gpu, 'Training')

  # Saving the Final model
  if save_state:
    state = {
              'epoch': epoch,
              'state_dict': net.state_dict(),
              'optimizer': optimizer_SGD.state_dict()
            }
    torch.save(state, PATH + time.strftime("%Y%m%d-%H%M%S") + '_' + str(epoch) + '.dat')


if __name__ == '__main__':
  main(*sys.argv[1:])
