import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import Audio_Frontend, MobileNetV2_Head, S3D_Head
from .fusions import CrossAttn_Fusion, SelfAttn_Fusion, Simple_CrossAttn_Fusion, MBT_Fusion
from .gate import LightweightCNN_Gate
from config import CONFIG

FUSION_MAP = {
    None: None,  # 2-branch
    'cross_attn': CrossAttn_Fusion,
    'self_attn': SelfAttn_Fusion,
    'simple_cross': Simple_CrossAttn_Fusion,
    'mbt': MBT_Fusion,
}

class DynamicFFIAModelOneSwitch(nn.Module):
    """
    One-switch inference model.
    Gate decides -> only ONE expert runs per sample.
    """
    def __init__(self, num_classes=4, fusion_type=None):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.num_experts = 2 if fusion_type is None else 3

        # Shared components (same names as training model)
        self.audio_frontend = Audio_Frontend(**CONFIG['audio_cfg'], training=False)
        self.audio_backbone = MobileNetV2_Head(output_dim=1024)
        self.video_backbone = S3D_Head(output_dim=1024)
        self.project_a = nn.Linear(1024, 512)
        self.project_v = nn.Linear(1024, 512)

        self.head_audio = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.head_video = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_classes))

        if fusion_type is not None:
            self.fusion_module = FUSION_MAP[fusion_type](embed_dim=512)   # same name as training
            self.head_fusion = nn.Linear(512, num_classes)

        self.gate = LightweightCNN_Gate(num_experts=self.num_experts, feature_dim=128)

    def forward(self, audio_raw, video_raw):
        """Gate decides → only ONE expert runs per sample (grouped)"""
        with torch.no_grad():
            spec = self.audio_frontend(audio_raw)                  # shared
            gate_logits = self.gate(spec, video_raw)
            chosen = torch.argmax(gate_logits, dim=1)              # (B,)

            B = audio_raw.shape[0]
            preds = torch.zeros(B, self.num_classes, device=audio_raw.device)

            for e in range(self.num_experts):
                mask = (chosen == e)
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                sub_spec = spec[idx]
                sub_video = video_raw[idx]

                if e == 0:   # Audio-only
                    a_feat_high = self.audio_backbone(sub_spec)
                    a_feat = F.relu(self.project_a(a_feat_high))
                    sub_pred = self.head_audio(a_feat.transpose(1, 2))
                elif e == 1: # Video-only
                    v_feat_high = self.video_backbone(sub_video)
                    v_feat = F.relu(self.project_v(v_feat_high))
                    sub_pred = self.head_video(v_feat.transpose(1, 2))
                else:        # Fusion (only if 3-branch)
                    a_feat_high = self.audio_backbone(sub_spec)
                    a_feat = F.relu(self.project_a(a_feat_high))
                    v_feat_high = self.video_backbone(sub_video)
                    v_feat = F.relu(self.project_v(v_feat_high))
                    fused = self.fusion_module(a_feat, v_feat)
                    sub_pred = self.head_fusion(fused)

                preds[idx] = sub_pred

            return preds, chosen