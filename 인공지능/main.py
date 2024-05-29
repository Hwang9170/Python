import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torchaudio
import logging
from copy import deepcopy
import csv
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict


### Utils

def find_wav_files(path_to_dir: Union[Path, str]) -> Optional[List[Path]]:
    """Find all wav files in the directory and its subtree.

    Args:
        path_to_dir: Path top directory.
    Returns:
        List containing Path objects or None (nothing found).
    """
    paths = list(sorted(Path(path_to_dir).glob("**/*.wav")))

    if len(paths) == 0:
        return None
    return paths


def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            directory_or_path_list: Union[Union[str, Path], List[Union[str, Path]]],
            sample_rate: int = 16_000,
            normalize: bool = True,
            real: str = 'real',
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.normalize = normalize
        if real == 'real':
            directory_or_path_list = directory_or_path_list * 5

        if isinstance(directory_or_path_list, list):
            paths = directory_or_path_list
        elif isinstance(directory_or_path_list, Path) \
                or isinstance(directory_or_path_list, str):
            directory = Path(directory_or_path_list)
            if not directory.exists():
                raise IOError(f"Directory does not exists: {directory}")

            paths = find_wav_files(directory)
            if paths is None:
                raise IOError(
                    f"Directory did not contain wav files: {directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!")


        self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if sample_rate != self.sample_rate:
            transform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = transform(waveform)


        return waveform, sample_rate

    def __len__(self) -> int:
        return len(self._paths)


class PadDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, cut: int = 64600, label=None):
        self.dataset = dataset
        self.cut = cut  # max 4 sec (ASVSpoof default)
        self.label = label

    def __getitem__(self, index):
        waveform, sample_rate = self.dataset[index]
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]
        if waveform_len >= self.cut:
            if self.label is None:
                return waveform[:self.cut], sample_rate
            else:
                return waveform[:self.cut], sample_rate, self.label
        # need to pad
        num_repeats = int(self.cut / waveform_len)+1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[
            :, :self.cut][0]

        if self.label is None:
            return padded_waveform, sample_rate
        else:
            return padded_waveform, sample_rate, self.label

    def __len__(self):
        return len(self.dataset)




def load_dataset(
        path: Union[Path, str],
        pad: bool = False,
        train: str = 'train',
        real: str = 'real',
        label: Optional[int] = None,
) -> torch.utils.data.Dataset:

    cur_path = "{}/{}/{}".format(path,train,real)
    
    paths = find_wav_files(cur_path)
    if paths is None:
        raise IOError(f"Could not load files from {path}!")

    LOGGER.info(f"Loading data from {path}...!")

    dataset = AudioDataset(
        paths, real=real)
    if pad:
        dataset = PadDataset(dataset, label=label)

    return dataset




def load_dataset_test(
        path: Union[Path, str],
        pad: bool = True,
) -> torch.utils.data.Dataset:


    paths = find_wav_files(path)
    if paths is None:
        raise IOError(f"Could not load files from {path}!")


    dataset = AudioDataset(
        paths, real='fake')
    if pad:
        dataset = PadDataset(dataset, label=0)

    return dataset

# raw_Net2

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv,self).__init__()

        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)
        fmel=self.to_mel(f)   # Hz to mel conversion
        fmelmax=np.max(fmel)
        fmelmin=np.min(fmel)
        filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
        filbandwidthsf=self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel=filbandwidthsf
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels,self.kernel_size)
    
       
        
    def forward(self,x):
        for i in range(len(self.mel)-1):
            fmin=self.mel[i]
            fmax=self.mel[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


        
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0]!=nb_filts[1]:
            self.downsample=nn.Conv1d(in_channels = nb_filts[0], out_channels = nb_filts[1], kernel_size = 1)
        else:
            self.downsample = None
        
        
    def forward(self, x):
        
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
            
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        
        return out

class RawNet2(nn.Module):
    def __init__(self, device, block, nb_blocks, nb_filts, first_conv, in_channels, sr, nb_samp, 
        filter_size=251, gru_node=1024, gru_layers=1, n_classes=2):
        super(RawNet2, self).__init__()
        
        self.first_conv = nn.Conv1d(in_channels = in_channels,
			out_channels = first_conv[0],
			kernel_size = first_conv[1],
			stride = first_conv[2])
        self.device=device

        self.firstbn = nn.BatchNorm1d(num_features = first_conv[0])
        
        self.sinc_conv=SincConv(device=device, out_channels=nb_filts[0][0], kernel_size=filter_size)

        self.first_block = self._make_layer(block, nb_filts[1], nb_blocks[0], first = True)
        self.block0 = self._make_layer(block, nb_filts[2], nb_blocks[1])
        self.block1 = self._make_layer(block, nb_filts[3], nb_blocks[2])
        self.block2 = self._make_layer(block, nb_filts[4], nb_blocks[3])
        self.block3 = self._make_layer(block, nb_filts[5], nb_blocks[4])
        
        self.bn_before_gru = nn.BatchNorm1d(num_features = nb_filts[5][-1])
        
        self.gru = nn.GRU(input_size = int(nb_samp/8*nb_filts[5][-1]),
			hidden_size = gru_node,
			num_layers = gru_layers,
			batch_first = True)
        
        self.fc1 = nn.Linear(in_features = gru_node,
			out_features = gru_node)
        
        self.fc2 = nn.Linear(in_features = gru_node,
			out_features = n_classes)
        
        self.lrelu = nn.LeakyReLU(0.3)
        
        
    def _make_layer(self, block, nb_filts, nb_blocks, first = False):
        layers = []
        layers.append(block(nb_filts, first = first))
        for i in range(1, nb_blocks):
            layers.append(block(nb_filts))
        
        return nn.Sequential(*layers)

        
    def forward(self, x):
        
        x=x.unsqueeze(1)
        x=x.to(self.device)

        x=self.sinc_conv(x)

        x = self.first_conv(x)
        x = self.firstbn(x)
        x = self.lrelu(x)
        x = self.first_block(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.bn_before_gru(x)
        x = self.lrelu(x)
        x = torch.flatten(x, 1)
        
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        
        out = out[:,-1,:]
        code = self.fc1(out)
        code = self.lrelu(code)
        x = self.fc2(code)
        
        return x, code

class GDOptimiser():
    def __init__(self, model, learning_rate, max_gradient, weight_decay, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.max_gradient = max_gradient

    def do_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_gradient > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient)
        self.optimizer.step()
        
class GDTrainer:
    def __init__(self, device, train_loader, model, optimiser, epoch, criterion):
        self.model = model
        self.optimiser = optimiser
        self.device = device
        self.train_loader = train_loader
        self.epoch = epoch
        self.criterion = criterion
        self.loss_epoch = 0

    def train(self):
        self.model.train()
        self.loss_epoch = 0

        for batch_idx, (data, label) in enumerate(self.train_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            output, code = self.model(data)
            loss = self.criterion(output, label)
            self.optimiser.do_step(loss)

            self.loss_epoch += loss.item()
        return self.loss_epoch / len(self.train_loader)


def train_raw_net(train_data, epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RawNet2(device, Residual_block, nb_blocks=[2, 2, 2, 2, 2], nb_filts=[
        [128], [128, 128], [128, 256], [256, 256], [256, 512], [512, 512]], first_conv=[128, 3, 3], in_channels=1, 
        sr=16000, nb_samp=64600, filter_size=251, gru_node=1024, gru_layers=1, n_classes=2)
    
    optimiser = GDOptimiser(model, learning_rate=0.001, max_gradient=2, weight_decay=0.0001, device=device)
    
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    
    trainer = GDTrainer(device, train_loader, model, optimiser, epoch=epochs, criterion=criterion)
    
    for epoch in range(epochs):
        loss = trainer.train()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return model


# Example usage
if __name__ == "__main__":
    set_seed(42)
    LOGGER = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    dataset_path = "path_to_dataset"
    train_data = load_dataset(dataset_path, pad=True, train='train', real='real', label=1)
    model = train_raw_net(train_data, epochs=10)
