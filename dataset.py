from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord:
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataset(Dataset):
    def __init__(self, root_path, list_file, num_segments=3, new_length=1,
                 modality='RGB', image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self._parse_list()

        if self.modality == 'RGBDiff':
            self.new_length += 1

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        param: record: VideoRecord
        return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                      randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def get(self, record, indices):
        images = []
        for seg_idx in indices:
            p = int(seg_idx)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        proces_data = self.transform(images)
        return proces_data, record.label

    def __getitem__(self, idx):
        record = self.video_list[idx]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import torchvision
    from transform import GroupScale, GroupRandomCrop, Stack, ToTorchFormatTensor, GroupNormalize

    root_path = 'data/RGB/'
    list_file = 'data_split/train_ucf.txt'
    transforms = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    dataset = TSNDataset(root_path, list_file, transform=transforms)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for i, (data, label) in enumerate(train_loader):
        print(data.shape)
        break
