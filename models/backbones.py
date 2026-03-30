import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.video import s3d
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class Audio_Frontend(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, training):
        super().__init__()
        self.training = training
        self.mel_bins = mel_bins
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
        )
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

    def forward(self, x):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        m = nn.ZeroPad2d((0, 0, 2, 0))
        x = m(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        return x

class MobileNetV2_Head(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        backbone = mobilenet_v2(weights='DEFAULT')
        first_conv = backbone.features[0][0]
        new_first_conv = nn.Conv2d(1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                                   stride=first_conv.stride, padding=first_conv.padding, bias=False)
        new_first_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_first_conv
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Linear(1280, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(3).transpose(1, 2)
        x = self.fc(x)
        return x

class S3D_Head(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        backbone = s3d(weights='DEFAULT')
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        B, C, T, H, W = x.shape
        x = x.squeeze(4).squeeze(3)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x