import os
import glob
import numpy as np
import torch
from itertools import chain
import librosa
from scipy.signal import resample
import decord
from torchvision import transforms
import dtk.transforms as dtf
from torch.utils.data import Dataset, DataLoader
from config import CONFIG

decord.bridge.set_bridge("torch")

def get_audio_file(video_files):
    dir, audio_name = os.path.split(video_files)
    audio_id = (audio_name.split('.')[0])[9:]
    dir1, audio_name1 = os.path.split(dir)
    dir2, audio_name2 = os.path.split(dir1)
    dir3, audio_name3 = os.path.split(dir2)
    audio_files = glob.glob(
        f"{CONFIG['data_root']}/audio_dataset/audio_dataset/{audio_name3}/{audio_name2}/{audio_name1}/*.wav"
    )
    for audio_file in audio_files:
        dir4, audio_name = os.path.split(audio_file)
        video_id = (audio_name.split('.')[0])[9:]
        if str(video_id) == str(audio_id):
            return os.path.join(dir4, audio_name)
    return None

def load_audio(path, sr=None):
    y, _ = librosa.load(path, sr=None)
    y = resample(y, num=sr * 2)
    return y

def get_video_name(split='strong'):
    root = CONFIG['data_root']
    video = []
    video_roots = [os.path.join(root, d) for d in os.listdir(root) if d.startswith('video_') and os.path.isdir(os.path.join(root, d))]
    for vr in video_roots:
        for dir1 in os.listdir(vr):
            p1 = os.path.join(vr, dir1)
            if not os.path.isdir(p1): continue
            for dir2 in os.listdir(p1):
                video_dir = os.path.join(p1, dir2, split, '*.mp4')
                video.append(glob.glob(video_dir))
    return list(chain.from_iterable(video))

def data_generator(seed, test_sample_per_class):
    """
    class to label mapping:
    none: 0
    strong: 1
    middle: 2
    weak: 3
    """
    random_state = np.random.RandomState(seed)
    strong_v_list = get_video_name(split='strong')
    medium_v_list = get_video_name(split='medium')
    weak_v_list = get_video_name(split='weak')
    none_v_list = get_video_name(split='none')

    random_state.shuffle(strong_v_list)
    random_state.shuffle(medium_v_list)
    random_state.shuffle(weak_v_list)
    random_state.shuffle(none_v_list)

    strong_v_train = strong_v_list[2 * test_sample_per_class :]
    medium_v_train = medium_v_list[2 * test_sample_per_class :]
    weak_v_train = weak_v_list[2 * test_sample_per_class :]
    none_v_train = none_v_list[2 * test_sample_per_class :]

    strong_v_test = strong_v_list[:test_sample_per_class]
    medium_v_test = medium_v_list[:test_sample_per_class]
    weak_v_test = weak_v_list[:test_sample_per_class]
    none_v_test = none_v_list[:test_sample_per_class]

    strong_v_val = strong_v_list[test_sample_per_class : 2 * test_sample_per_class]
    medium_v_val = medium_v_list[test_sample_per_class : 2 * test_sample_per_class]
    weak_v_val = weak_v_list[test_sample_per_class : 2 * test_sample_per_class]
    none_v_val = none_v_list[test_sample_per_class : 2 * test_sample_per_class]

    train_v_dict = []
    test_v_dict = []
    val_v_dict = []

    for wav in strong_v_train:
        train_v_dict.append([wav, 1])

    for wav in medium_v_train:
        train_v_dict.append([wav, 2])

    for wav in weak_v_train:
        train_v_dict.append([wav, 3])

    for wav in none_v_train:
        train_v_dict.append([wav, 0])

    for wav in strong_v_test:
        test_v_dict.append([wav, 1])

    for wav in medium_v_test:
        test_v_dict.append([wav, 2])

    for wav in weak_v_test:
        test_v_dict.append([wav, 3])

    for wav in none_v_test:
        test_v_dict.append([wav, 0])

    for wav in strong_v_val:
        val_v_dict.append([wav, 1])

    for wav in medium_v_val:
        val_v_dict.append([wav, 2])

    for wav in weak_v_val:
        val_v_dict.append([wav, 3])

    for wav in none_v_val:
        val_v_dict.append([wav, 0])

    random_state.shuffle(train_v_dict)
    random_state.shuffle(test_v_dict)
    random_state.shuffle(val_v_dict)

    return train_v_dict, test_v_dict, val_v_dict

class Fish_Video_Dataset(Dataset):
    def __init__(self, seed, epoch, sample_rate, split):
        """
        split: train or test
        if sample_rate=None, read audio with the default sr
        """
        self.seed = seed
        self.sample_rate = sample_rate
        self.split = split
        train_v_dict, test_v_dict, val_v_dict = data_generator(
            self.seed, test_sample_per_class=700
        )
        #train_v_dict, test_v_dict, val_v_dict = data_generator(
        #    self.seed, test_sample_per_class=5
        #)
        if self.split == 'train':
            self.data_dict = train_v_dict
        elif self.split == 'test':
            self.data_dict = test_v_dict
        elif self.split == 'val':
            self.data_dict = val_v_dict
        self.epoch = epoch

        transform_chain = [dtf.ToTensorVideo()]
        #transform_chain += [dtf.NormalizeVideo([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        #transform_chain += [CenterCropVideo(196)]
        # transform_chain += [RandomHorizontalFlipVideo(0.5)]

        # TODO add data augmentation from dtk
        self.video_transform = transforms.Compose(transform_chain)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        video_name, target = self.data_dict[index]
        target = np.eye(4)[target]
        # random sample one frame from video
        vr = decord.VideoReader(video_name, height=224, width=224)
        full_vid_length = len(vr)
        video_frames = vr.get_batch(range(0, full_vid_length))

        X = np.arange(0, full_vid_length - 1)
        Y = sorted(np.random.choice(X, 20, replace=False))
        vf = video_frames[Y, ...]
        vf = self.video_transform(vf)
        wav_name = get_audio_file(video_name)
        wav = load_audio(wav_name, sr=self.sample_rate)
        wav = np.array(wav)
        data_dict = {'video_name': video_name, 'video_form': vf, 'target': target, 'audio_name': wav_name, 'waveform': wav}
        return data_dict

def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]
    wav = torch.FloatTensor(np.array(wav))
    video_name = [data['video_name'] for data in batch]
    vf = torch.stack([data['video_form'] for data in batch])
    vf_all = torch.stack([data['video_form'] for data in batch])
    # vf = vf.squeeze(1)
    vf = vf.permute(0, 2, 1, 3, 4)
    vf_all = vf_all.permute(0, 2, 1, 3, 4)
    target = torch.FloatTensor(np.array(target))
    data_dict = {
        'video_name': video_name,
        'video_form': vf,
        'video_form_all': vf_all,
        'target': target,
        'audio_name': wav_name,
        'waveform': wav,
    }

    return data_dict

def get_dataloader(split, batch_size, sample_rate, seed, shuffle=False, drop_last=False, epoch=0, num_workers=0):
    dataset = Fish_Video_Dataset(split=split, seed=seed, sample_rate=sample_rate, epoch=epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn)