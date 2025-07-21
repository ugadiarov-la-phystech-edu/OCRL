import time

import numpy as np
import skimage.io
import torchvision.transforms
from torch.utils.data import Dataset
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EpisodesDataset(Dataset):
    def __init__(self, root, mode, obs_size=128, allow_resize=True, extension='png', return_tensor=True, kind='image',
                 sequence_length=1, augmentation_probability=0.):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode in ('valid', 'test'):
            mode = 'val'

        assert kind in ('image', 'video'), f'Expected kind: image or video. Actual: {kind}'
        if kind == 'image':
            assert sequence_length == 1, f'Expected sequence length: 1. Actual: {sequence_length}'

        self.kind = kind
        self.sequence_length = sequence_length

        root = os.path.join(root, mode)
        root_with_obs = os.path.join(root, 'obs')
        if os.path.isdir(root_with_obs):
            self.root = root_with_obs
        else:
            self.root = root

        self.res = obs_size
        self.mode = mode
        self.extension = extension
        self.return_tensor = return_tensor
        self.augmentation_probability = augmentation_probability
        self.to_tensor = transforms.ToTensor()
        self.augmentation_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomResizedCrop(size=self.res, scale=(0.4, 1.), ratio=(1, 1)),
             transforms.RandomHorizontalFlip(), ])

        # Get all numbers
        self.folders = []
        start = time.time()
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue

        def get_num(x):
            parts = x.split('_')
            num = parts[0] if len(parts) == 1 else parts[1]
            return int(num)

        self.folders.sort(key=get_num)

        self.episode_images = []
        self.episode2offset = [0]
        self.index2episode = []
        for i, f in enumerate(self.folders):
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, f'*.{self.extension}')))
            actual_length = len(paths)
            get_file_id = lambda x: get_num(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_file_id)
            self.episode_images.append(paths)
            self.index2episode.extend([len(self.episode_images) - 1] * actual_length)
            self.episode2offset.append(self.episode2offset[-1] + actual_length)

        print(f'Dataset indexing took {time.time() - start} seconds')

    def __getitem__(self, index):
        if self.kind == 'video':
            episode_images = self.episode_images[index]
            start_index = np.random.randint(0, len(episode_images) - self.sequence_length + 1)

            image_sequence = []
            for image_index in range(start_index, start_index + self.sequence_length):
                img = Image.open(episode_images[image_index])
                img = img.resize((self.res, self.res))
                image_sequence.append(img)

            if self.return_tensor:
                image_sequence = torch.stack([self.to_tensor(img) for img in image_sequence], dim=0)

            return {'obss': image_sequence}
        elif self.kind == 'image':
            ep = self.index2episode[index]
            # Implement continuous indexing
            offset = self.episode2offset[ep]
            in_episode_index = index - offset
            img = skimage.io.imread(self.episode_images[ep][in_episode_index])
            if np.random.random() < self.augmentation_probability:
                center = (np.random.random() * img.shape[1], np.random.random() * img.shape[0])
                img = skimage.transform.rotate(img, angle=np.random.random() * 360, mode='reflect', center=center)
                img = skimage.util.img_as_ubyte(img)
                img = self.augmentation_transforms(img)
            else:
                img = Image.open(self.episode_images[ep][in_episode_index])
                img = img.resize((self.res, self.res))
                img = self.to_tensor(img)

            if not self.return_tensor:
                img = np.array(torchvision.transforms.ToPILImage(img))

            return {'obss': img}
        else:
            assert False, 'Cannot happen!'

    def __len__(self):
        if self.kind == 'image':
            return len(self.index2episode)
        elif self.kind == 'video':
            return len(self.episode_images)
        else:
            assert False, 'Cannot happen!'
