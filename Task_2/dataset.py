from torch.utils.data import Dataset, DataLoader
import os
import torch
from torch.utils.data import Dataset
import numpy as np



class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Loop through clean folder to get file paths
        # Create a list of all clean spectrogram paths in the dataset
        self.clean_files = []
        clean_dir = os.path.join(data_dir, 'clean')
        for speaker_dir in os.listdir(clean_dir):
            speaker_path = os.path.join(clean_dir, speaker_dir)
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                self.clean_files.append(file_path)

        # Create a list of all noise spectrogram paths in the dataset
        self.noise_files = []
        noise_dir = os.path.join(data_dir, 'noisy')
        for speaker_dir in os.listdir(noise_dir):
            speaker_path = os.path.join(noise_dir, speaker_dir)
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                self.noise_files.append(file_path)

    def __getitem__(self, index):
        # Read clean and noisy spectrograms from files
        clean_spec = torch.from_numpy(np.load(self.clean_files[index]))
        noise_spec = torch.from_numpy(np.load(self.noise_files[index]))
        noise_spec = noise_spec[None, :, :]
        clean_spec = clean_spec[None, :, :]
        return clean_spec, noise_spec


    def __len__(self):
        return len(self.clean_files)


def _group_random_crop(img_group, crop_height, crop_width):
    if len(img_group) == 0:
        return ()
    else:

        _, height, width = img_group[0].shape

        random_height = (
            0 if height == crop_height else round((height - crop_height)/2)
        )


        random_width = (
            0 if width == crop_width else round((width - crop_width) / 2)
        )

        return tuple(
            image[:,
                random_height: random_height + crop_height,
                random_width: random_width + crop_width,
            ]
            for image in img_group
        )



def _collate_with_cropping(batch):

    crop_height = min([item[0].shape[1] for item in batch])
    crop_width = min([item[0].shape[2] for item in batch])

    cropped_img_groups = [
        _group_random_crop(img_group, crop_height, crop_width) for img_group in batch
    ]

    return tuple(map(torch.stack, zip(*cropped_img_groups)))


def load(
    dirname,
    batch_size = 2,
    collate_fn= _collate_with_cropping,
    validate = False
):

    dataset = CustomDataset(dirname)
    if not validate:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    return loader
