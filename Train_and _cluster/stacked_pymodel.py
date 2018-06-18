import argparse
import os
import random
import warnings
from glob import glob

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from utils import strided_app
import math
from time import localtime, strftime

# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

class MyNet(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()

        c1 = nn.Sequential(
            nn.Conv1d(1, 10, 5, padding=4),
            nn.MaxPool1d(2), nn.SELU())

        c2 = nn.Sequential(
            nn.Conv1d(10, 15, 3, padding=2, dilation=2),
            nn.MaxPool1d(2), nn.SELU())

        c3 = nn.Sequential(
            nn.Conv1d(15, 15, 3, padding=4, dilation=4),
            nn.MaxPool1d(2), nn.SELU())

        self.feature_extractor = nn.Sequential(c1, c2, c3)

        self.c4 = nn.Sequential(
            nn.Conv1d(60, 30, 3, padding=4, dilation=4),
            nn.SELU())

        self.classifier = nn.Sequential(
            nn.Linear(150, 50), nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(50, 10), nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(10, 1))

        self.regressor = nn.Sequential(
            nn.Linear(150, 50), nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(50, 10), nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(10, 1),
            nn.Hardtanh(0, 40))

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight.data)
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.out_channels
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x.size())
        x = x.view(x.size(0), -1, np.int32(x.size(2)/4.0))
        # print("Mohit")
        x = self.c4(x)
        x = x.view(x.size(0), -1)
        x2 = self.regressor(x)
        x1 = self.classifier(x)
        y = th.squeeze(th.cat((x1, x2), 1))
        return y


def custom_loader(speechfolder, peakfolder, window, stride, subwindow=40):
    speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))
    peakfiles = sorted(glob(os.path.join(peakfolder, '*.npy')))

    speech_data = [np.load(f) for f in speechfiles]
    peak_data = [np.load(f) for f in peakfiles]

    speech_data = np.concatenate(speech_data)
    peak_data = np.concatenate(peak_data)

    speech_windowed_data = strided_app(speech_data, window, stride)
    peak_windowed_data = strided_app(peak_data, window, stride)

    peak_distance = np.array([
        np.nonzero(t)[0][0] if len(np.nonzero(t)[0]) != 0 else -1
        for t in peak_windowed_data
    ])

    ######
    # print(peak_distance.shape)
    ind = np.logical_and(peak_distance >=0, peak_distance <= subwindow) * 1.0
    print(np.sum(ind))

    # peak_indicator = (peak_distance != -1) * 1.0
    peak_indicator = ind

    return speech_windowed_data, peak_distance, peak_indicator


def create_dataloader(batch_size, speechfolder, peakfolder, window, stride):
    speech_windowed_data, peak_distance, peak_indicator = custom_loader(
        speechfolder, peakfolder, window, stride)
    peak_dataset = np.column_stack((peak_distance, peak_indicator))
    dataset = TensorDataset(
        th.from_numpy(
            np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
        th.from_numpy(peak_dataset.astype(np.float32)))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True)
    return dataloader


def train(model: nn.Module,
          optimizer: optim.Optimizer,
          train_data: DataLoader,
          use_cuda: bool = True,
          scheduler=None,
          bce_weight: float = 1,
          mse_weight: float = 0.1,
          misclass_weight: float = 1,
          corclass_weight: float = 1 ,
          threshold: float = 0.7,
          gci_threshold: float = 0.5):
    model.train()
    loss_sum = 0
    bce_loss = 0
    mse_loss = 0
    gci_misclass = 0
    misses = 0
    bce_weight = Variable(th.Tensor([bce_weight]))
    mse_weight = Variable(th.Tensor([mse_weight]))
    misclass_weight = Variable(th.Tensor([misclass_weight]))
    corclass_weight = Variable(th.Tensor([corclass_weight]))
    thresh = Variable(th.Tensor([threshold]))
    gci_thresh = Variable(th.Tensor([gci_threshold]))
    batches = len(train_data)

    if use_cuda:
        if th.cuda.is_available():
            model.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')
    for data, target in train_data:
        if scheduler is not None:
            scheduler.step()

        if use_cuda:
            data, target = data.cuda(), target.cuda()
            bce_weight = bce_weight.cuda()
            mse_weight = mse_weight.cuda()
            misclass_weight = misclass_weight.cuda()
            corclass_weight = corclass_weight.cuda()
            thresh_val = thresh.cuda()
            gci_thresh = gci_thresh.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # print(len(data))

        peak_distance_target = target[:, 0]
        peak_indicator_target = target[:, 1]
        output = model(data)

        distance = (output[:, 1])
        probabilities = output[:, 0]

        loss_bce = F.binary_cross_entropy_with_logits(probabilities,
                                                      peak_indicator_target)
        # print(loss_bce, loss_bce.mean())
        loss_mse = (distance * peak_indicator_target - peak_distance_target * peak_indicator_target) ** 2
        loss_mse = loss_mse.sum()/peak_indicator_target.sum()
        out = (F.sigmoid(probabilities) > gci_thresh).float()
        loss_misclass = (1 - peak_indicator_target) * (
            F.sigmoid(probabilities)**2)
        # loss_misclass = (1 - peak_indicator_target) * (out)
        loss_misclass = loss_misclass.mean()

        misses_temp = (1 - peak_indicator_target) * out
        misses += misses_temp.mean().data[0]

        out = (F.sigmoid(probabilities) > gci_thresh).float()
        gci_misclass_temp = peak_indicator_target * (1 - out)
        gci_misclass += gci_misclass_temp.mean().data[0]

        loss_corrclass = peak_indicator_target * ((
            1 - F.sigmoid(probabilities))**2)
        loss_corrclass = loss_corrclass.mean()

        net_loss = bce_weight * loss_bce + mse_weight * loss_mse

        loss_sum += net_loss.data[0]
        bce_loss += loss_bce.data[0]
        mse_loss += loss_mse.data[0]

        net_loss.backward()
        # TODO: Gradient Clipping
        optimizer.step()
    return loss_sum / batches , bce_loss / batches , mse_loss / batches , gci_misclass / batches, misses / batches


def test(model: nn.Module,
         test_loader: DataLoader,
         use_cuda: bool = False,
         threshold: float = 0.5):
    if use_cuda:
        model.cuda()
    model.eval()
    bce_loss = 0
    mse_loss = 0
    eval_misclass = 0
    gci_misclass = 0
    batches = len(test_loader)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        batch_size = len(data)
        output = model(data)

        peak_distance_target = target[:, 0]
        peak_indicator_target = target[:, 1]

        # sum up batch loss

        distance = output[:, 1]
        probabilities = output[:, 0]

        bce_loss += F.binary_cross_entropy_with_logits(probabilities,
                                                      peak_indicator_target)

        loss_mse = (distance * peak_indicator_target - peak_distance_target * peak_indicator_target) ** 2
        loss_mse = loss_mse.sum()/peak_indicator_target.sum()

        mse_loss += loss_mse

        out = (F.sigmoid(probabilities) > threshold).float()
        loss_misclass = (1 - peak_indicator_target) * out
        loss_misclass = loss_misclass.mean()
        eval_misclass += loss_misclass.data[0]
        gci_misclass += (peak_indicator_target * (1 - out)).mean().data[0]

    mean_misclass = eval_misclass / batches
    mse_loss /= batches  # type: ignore
    mse_loss = mse_loss.data[0]
    gci_misclass = gci_misclass / batches
    bce_loss /= batches
    print(
        '\nTest set: MSE loss: {:.4f}, BCE loss: {:.4f} Mean Batch Misclassification {:.4f} GCI Misses {:.4f}\n'.
        format(mse_loss, bce_loss, mean_misclass, gci_misclass))


def main():
    train_data = create_dataloader(128, 'apld_train/speech', 'apld_train/peaks', 160, 3)
    test_data = create_dataloader(128, 'apld_test/speech', 'apld_test/peaks', 160, 20)
    model = MyNet()
    save_model = Saver('checkpoints/bce_slow')
    use_cuda = True
    epochs = 16

    optimizer = optim.Adamax(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in range(epochs):

            nloss, bloss, mloss, gci, miss = train(model, optimizer,
                                                   train_data, use_cuda)
            print(
                'Train Net Loss: {:.4f} BCE Loss {:.4f} MSE Loss {:.4f} GCI {:.4f} Misclass {:.4f} @epoch {}'.
                format(nloss, bloss, mloss, gci, miss, i))

            if i % 5 == 0:
                test(model, test_data, use_cuda)
                checkpoint = save_model.create_checkpoint(model, optimizer, {'win': 160, 'stride': 5 })
                save_model.save_checkpoint(checkpoint, file_name='bce_epoch_{}.pt'.format(i), append_time=False)
            if scheduler is not None:
                scheduler.step()

class Saver:
    def __init__(self, directory: str = 'pytorch_model',
                 iteration: int = 0) -> None:
        self.directory = directory
        self.iteration = iteration

    def save_checkpoint(self,
                        state,
                        file_name: str = 'pytorch_model.pt',
                        append_time=True):
        os.makedirs(self.directory, exist_ok=True)
        timestamp = strftime("%Y_%m_%d__%H_%M_%S", localtime())
        filebasename, fileext = file_name.split('.')
        if append_time:
            filepath = os.path.join(self.directory, '_'.join(
                [filebasename, '.'.join([timestamp, fileext])]))
        else:
            filepath = os.path.join(self.directory, file_name)
        if isinstance(state, nn.Module):
            checkpoint = {'model_dict': state.state_dict()}
            th.save(checkpoint, filepath)
        elif isinstance(state, dict):
            th.save(state, filepath)
        else:
            raise TypeError('state must be a nn.Module or dict')

    def load_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer = None,
                        file_name: str = 'pytorch_model.pt'):
        filepath = os.path.join(self.directory, file_name)
        checkpoint = th.load(filepath)
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])

        hyperparam_dict = {
            k: v
            for k, v in checkpoint.items()
            if k != 'model_dict' or k != 'optimizer_dict'
        }

        return model, optimizer, hyperparam_dict

    def create_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                          hyperparam_dict):
        model_dict = model.state_dict()
        optimizer_dict = optimizer.state_dict()

        state_dict = {
            'model_dict': model_dict,
            'optimizer_dict': optimizer_dict,
            'timestamp': strftime('%l:%M%p GMT%z on %b %d, %Y', localtime())
        }
        checkpoint = {**state_dict, **hyperparam_dict}

        return checkpoint



if __name__ == "__main__":
    main()
