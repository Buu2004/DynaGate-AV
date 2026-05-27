import torch
import torch.nn as nn
import torch.nn.functional as F

def DiffSoftmax(logits, tau=1.0, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
   
    return ret

class LightweightCNN_Gate(nn.Module):
    def __init__(self, audio_channels=1, video_channels=3, feature_dim=128, num_experts=2):
        super().__init__()
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

class LightweightTransformer_Gate(nn.Module):
    def __init__(self, audio_channels=1, video_channels=3, embed_dim=64, num_heads=4, num_experts=2):
        super().__init__()
        
        self.audio_patcher = nn.Sequential(
            nn.Conv2d(audio_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        self.video_patcher = nn.Sequential(
            nn.Conv2d(video_channels, 32, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        
        self.gate_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_experts) 
        )

    def forward(self, audio_spec, video_frames):
        B = audio_spec.size(0)
        
        a_patches = self.audio_patcher(audio_spec) 
        a_tokens = a_patches.flatten(2).transpose(1, 2) 
        
        v_collapsed = video_frames.mean(dim=2) 
        v_patches = self.video_patcher(v_collapsed) 
        v_tokens = v_patches.flatten(2).transpose(1, 2) 
        
        gate_tokens = self.gate_token.expand(B, -1, -1)
        
        x = torch.cat([gate_tokens, a_tokens, v_tokens], dim=1)
        
        x = self.transformer(x)
        
        gate_out = x[:, 0, :] 
        
        logits = self.mlp(gate_out)
        
        return logits


class LightweightMLP_Gate(nn.Module):
    def __init__(
        self,
        audio_feat_dim=128,
        video_feat_dim=128,
        hidden_dim=128,
        audio_pool=(8, 16),   # pooled audio grid: (time_bins, freq_bins)
        video_pool=(4, 4),    # pooled video grid: (spatial_h, spatial_w)
        num_experts=2
    ):
        super().__init__()

        self.audio_pool = audio_pool
        self.video_pool = video_pool

        audio_in_dim = 1 * audio_pool[0] * audio_pool[1]
        video_in_dim = 3 * video_pool[0] * video_pool[1]

        self.audio_proj = nn.Linear(audio_in_dim, hidden_dim)
        self.video_proj = nn.Linear(video_in_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, audio_spec, video_frames):
        a_pooled = F.adaptive_avg_pool2d(audio_spec, self.audio_pool)
        a_pooled = a_pooled.flatten(1)
        a_feat = self.audio_proj(a_pooled)

        v_temporal = video_frames.mean(dim=2)
        v_pooled = F.adaptive_avg_pool2d(v_temporal, self.video_pool)
        v_pooled = v_pooled.flatten(1)
        v_feat = self.video_proj(v_pooled)

        combined = torch.cat([a_feat, v_feat], dim=1)
        logits = self.mlp(combined)

        return logits