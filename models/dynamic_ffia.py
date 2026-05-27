import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import Audio_Frontend, MobileNetV2_Head, S3D_Head
from .fusions import CrossAttn_Fusion, SelfAttn_Fusion, Simple_CrossAttn_Fusion, MBT_Fusion
from .gate import LightweightCNN_Gate, LightweightTransformer_Gate, LightweightMLP_Gate, DiffSoftmax
from config import CONFIG

FUSION_MAP = {
    'cross_attn': CrossAttn_Fusion,
    'self_attn': SelfAttn_Fusion,
    'simple_cross': Simple_CrossAttn_Fusion,
    'mbt': MBT_Fusion,
}

FLOP_WEIGHTS = {
    None: torch.Tensor([0.5945, 22.5086]),                     # 2-branch
    'cross_attn': torch.Tensor([0.5945, 22.5086, 23.1284]),
    'simple_cross': torch.Tensor([0.5945, 22.5086, 23.1063]),
    'mbt': torch.Tensor([0.5945, 22.5086, 23.2292]),
    'self_attn': torch.Tensor([0.5945, 22.5086, 23.1126]),
}

class DynamicFFIAModel(nn.Module):
    def __init__(self, num_classes=4, tau=1.0, fusion_type=None, training=False):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.tau = tau
        self.num_experts = 2 if fusion_type is None else 3

        # Backbones
        self.audio_frontend = Audio_Frontend(**CONFIG['audio_cfg'], training=training)
        self.audio_backbone = MobileNetV2_Head(output_dim=1024)
        self.video_backbone = S3D_Head(output_dim=1024)

        self.project_a = nn.Linear(1024, 512)
        self.project_v = nn.Linear(1024, 512)

        # Heads
        self.head_audio = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.head_video = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_classes))

        if fusion_type is not None:
            self.fusion_module = FUSION_MAP[fusion_type](embed_dim=512)
            self.head_fusion = nn.Linear(512, num_classes)

        # Gate
        self.gate = LightweightCNN_Gate(num_experts=self.num_experts)

        self.weight_list = []
        self.store_weight = False
        self.flop_weights = FLOP_WEIGHTS[fusion_type].to(CONFIG['device'])

    def reset_weight(self):
        self.weight_list = []
        self.store_weight = True

    def weight_stat(self):
        if not self.weight_list:
            return tuple([0.0] * self.num_experts)
        all_weights = torch.cat(self.weight_list, dim=0)
        mean_weights = torch.mean(all_weights, dim=0).cpu()
        return tuple(mean_weights.tolist())

    def cal_flop(self):
        if not self.weight_list:
            return 0.0
        all_weights = torch.cat(self.weight_list, dim=0)
        mean_weights = torch.mean(all_weights, dim=0).cpu()
        flop_weights = self.flop_weights.to(mean_weights.device)
        total_flop = (flop_weights * mean_weights).sum().item()
        return total_flop

    def return_probs(self, weights):
        return weights + 1e-8

    def forward(self, audio_raw, video_raw, return_reg=False):
        spec = self.audio_frontend(audio_raw)
        a_feat_high = self.audio_backbone(spec)
        a_feat = F.relu(self.project_a(a_feat_high))

        v_feat_high = self.video_backbone(video_raw)
        v_feat = F.relu(self.project_v(v_feat_high))

        pred_a = self.head_audio(a_feat.transpose(1, 2))
        pred_v = self.head_video(v_feat.transpose(1, 2))

        if self.fusion_type is None:
            preds = torch.stack([pred_a, pred_v], dim=1)
        else:
            fused_rep = self.fusion_module(a_feat, v_feat)
            pred_f = self.head_fusion(fused_rep)
            preds = torch.stack([pred_a, pred_v, pred_f], dim=1)

        gate_logits = self.gate(spec, video_raw)
        weights = DiffSoftmax(gate_logits, tau=self.tau, dim=1)

        if self.store_weight:
            self.weight_list.append(weights.detach().cpu())

        output = (weights.unsqueeze(2) * preds).sum(dim=1)

        if return_reg:
            probs = self.return_probs(weights)
            return output, probs
        return output