import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode

import utils.webdataset.transforms
import utils.webdataset.preprocessing
from utils.webdataset.datasets import WebdatasetDataModule


def _image2tensor(image):
    return torch.as_tensor(image).permute(2, 0, 1)


def _normalize_float32(tensor):
    return tensor / 255.


def webdataset_dataloaders(config, batch_size, num_workers, shuffle_val):
    resize = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(_image2tensor),
        torchvision.transforms.Resize(config.obs_size, interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.Lambda(_normalize_float32),
    ])

    # keys_to_drop = ['__key__', 'image-filename', 'batch_size']
    transforms = {
        '00': utils.webdataset.transforms.SimpleTransform(transforms={'image': resize}, batch_transform=False),
        '01': utils.webdataset.transforms.Map(transform=utils.webdataset.preprocessing.RenameFields({'image': 'obss'}),
                                              fields=('image',), batch_transform=True),
        # '02': utils.webdataset.transforms.Map(transform=utils.webdataset.preprocessing.DropEntries(keys=keys_to_drop),
        #                                       fields=tuple(keys_to_drop), batch_transform=False)
    }
    wd = WebdatasetDataModule(
        train_shards=config.train_shards,
        val_shards=config.val_shards,
        train_size=config.train_size,
        val_size=config.val_size,
        train_transforms=transforms,
        eval_transforms=transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_val=shuffle_val
    )

    return wd.train_dataloader(), wd.val_dataloader()
