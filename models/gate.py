import torch
import torch.nn as nn

def DiffSoftmax(logits, tau=1.0, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
   
    return ret

class LightweightCNN_Gate(nn.Module):
    def __init__(self, audio_channels=1, video_channels=3, feature_dim=128, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        # Audio branch
        self.audio_net = nn.Sequential(
            nn.Conv2d(audio_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, feature_dim)
        )
        # Video branch
        self.video_net = nn.Sequential(
            nn.Conv2d(video_channels, 16, 5, stride=4, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, feature_dim)
        )
        # Decision MLP
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, audio_spec, video_frames):
        a_feat = self.audio_net(audio_spec)
        v_collapsed = video_frames.mean(dim=2)
        v_feat = self.video_net(v_collapsed)
        combined = torch.cat([a_feat, v_feat], dim=1)
        return self.mlp(combined)