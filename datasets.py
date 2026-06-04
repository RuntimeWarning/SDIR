import h5py
import torch
import datetime
import numpy as np
from torch.utils import data
from torchvision import transforms
from helpers.dataset_sevir import SEVIRTorchDataset



class Shanghai_Datasets(data.Dataset):
    """Shanghai radar sequence dataset backed by an HDF5 file."""

    def __init__(self, data_path, img_size, mode='train', trans=None):
        super().__init__()
        self.data_path = data_path
        self.img_size = img_size
        self.mode = mode if mode != 'val' else 'test'

        with h5py.File(data_path, 'r') as f:
            self.all_len = int(f[self.mode]['all_len'][()])
            
        self.file = None
        self.transform = trans or transforms.Compose([transforms.Resize((img_size, img_size))])

    def __len__(self):
        return self.all_len

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
        
        imgs = self.file[self.mode][str(index)][()]
        frames = torch.from_numpy(imgs).float() / 255.0
        return self.transform(frames.unsqueeze(1))


class CIKM_Datasets(data.Dataset):
    """CIKM radar sequence dataset backed by an HDF5 file."""

    def __init__(self, path, mode='train'):
        self.path = path
        self.mode = mode
        
        with h5py.File(self.path, 'r') as f:
            self.size = f[self.mode].shape[0]
            
        self.file = None
        self.dataset = None
        self.transform = transforms.CenterCrop((128, 128))

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
            self.dataset = self.file[self.mode]
        
        data = torch.from_numpy(self.dataset[index]).float() / 255.0
        data = self.transform(data)
        return data.unsqueeze(1)

    def __len__(self):
        return self.size
    

def get_datasets(name='cikm', opt='train', batch_size=16, num_workers=4, shuffle=True):
    """Create the requested dataset loader for train, test, or validation splits."""

    if name == 'cikm':
        cikm_dataset = CIKM_Datasets(path='../PN_Datasets/CIKM2017.h5', mode=opt)
        cikm_input_handle = data.DataLoader(cikm_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=shuffle)
        return cikm_input_handle

    elif name == 'shanghai':
        shanghai_dataset = Shanghai_Datasets(data_path='../PN_Datasets/shanghai.h5', img_size=256, mode=opt)
        shanghai_input_handle = data.DataLoader(shanghai_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=shuffle)
        return shanghai_input_handle

    elif name == 'sevir':
        train_valid_split = (2019, 1, 1)
        valid_test_split = (2019, 6, 1)
        if opt == 'train':
            train = SEVIRTorchDataset(
                dataset_dir='/data/zyl_data/SEVIR',
                split_mode='uneven',
                img_size=256,
                shuffle=shuffle,
                seq_len=25,
                stride=7,
                sample_mode='sequent',
                batch_size=batch_size,
                num_shard=1,
                rank=0,
                start_date=None,
                end_date=datetime.datetime(*train_valid_split),
                output_type=np.float32,
                preprocess=True,
                rescale_method='01',
                verbose=False
            )
            return train.get_torch_dataloader(num_workers=num_workers)
        elif opt == 'validation':
            val = SEVIRTorchDataset(
                dataset_dir='/data/zyl_data/SEVIR',
                split_mode='uneven',
                img_size=256,
                shuffle=shuffle,
                seq_len=25,
                stride=7,
                sample_mode='sequent',
                batch_size=batch_size,
                num_shard=1,
                rank=0,
                start_date=datetime.datetime(*train_valid_split),
                end_date=datetime.datetime(*valid_test_split),
                output_type=np.float32,
                preprocess=True,
                rescale_method='01',
                verbose=False
            )
            return val.get_torch_dataloader(num_workers=num_workers)
        else:
            test = SEVIRTorchDataset(
                dataset_dir='/data/zyl_data/SEVIR',
                split_mode='uneven',
                shuffle=shuffle,
                img_size=256,
                seq_len=25,
                stride=7,
                sample_mode='sequent',
                batch_size=batch_size,
                num_shard=1,
                rank=0,
                start_date=datetime.datetime(*valid_test_split),
                end_date=None,
                output_type=np.float32,
                preprocess=True,
                rescale_method='01',
                verbose=False
            )
            return test.get_torch_dataloader(num_workers=num_workers)
    else:
        raise ValueError('Unknown dataset name: {}'.format(name))
