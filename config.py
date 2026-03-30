import os
import random
import numpy as np
import torch

CONFIG = {
    'seed': 42,
    'batch_size': 22,
    'lr': 1e-5,
    'n_runs': 2,
    'n_epochs': 200,
    'num_classes': 4,
    'reg': 0.05,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'audio_cfg': {
        'sample_rate': 64000,
        'window_size': 2048,
        'hop_size': 1024,
        'mel_bins': 64,
        'fmin': 1,
        'fmax': 128000
    },
    'img_size': 224,
    'num_frames': 16,
    # Data paths (change these for your machine / dataset)
    'data_root': 'dataset/',  
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])
print(f"Running on {CONFIG['device']}")