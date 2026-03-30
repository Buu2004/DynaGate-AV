import torch
import torch.nn as nn

class CrossAttn_Fusion(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=512, dropout=0.1):
        super().__init__()
        self.att_linear = nn.Linear(input_dim, embed_dim)
        self.video_slf_attn = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.audio_slf_attn = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.av_slf_attn = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.video_cross_attn = nn.MultiheadAttention(embed_dim, 8, dropout=dropout, batch_first=True)
        self.audio_cross_attn = nn.MultiheadAttention(embed_dim, 8, dropout=dropout, batch_first=True)
        self.final_cross_attn = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, audio_feat, video_feat):
        av_feat = torch.cat((audio_feat, video_feat), dim=1)
        video_out, _ = self.video_slf_attn(video_feat, video_feat, video_feat)
        audio_out, _ = self.audio_slf_attn(audio_feat, audio_feat, audio_feat)
        av_out, _ = self.av_slf_attn(av_feat, av_feat, av_feat)
        video_context_out, _ = self.video_cross_attn(query=video_out, key=av_out, value=av_out)
        audio_context_out, _ = self.audio_cross_attn(query=audio_out, key=av_out, value=av_out)
        final_out, _ = self.final_cross_attn(query=audio_context_out, key=video_context_out, value=video_context_out)
        fused_mean = torch.mean(final_out, dim=1)
        return fused_mean

class SelfAttn_Fusion(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.1):
        super().__init__()
        self.video_slf_attn = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.audio_slf_attn = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8, dropout=dropout, batch_first=True)

    def forward(self, audio_feat, video_feat):
        video_out, _ = self.video_slf_attn(video_feat, video_feat, video_feat)
        audio_out, _ = self.audio_slf_attn(audio_feat, audio_feat, audio_feat)
        fused_out, _ = self.cross_attn(query=video_out, key=audio_out, value=audio_out)
        return torch.mean(fused_out, dim=1)

class Simple_CrossAttn_Fusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, audio_feat, video_feat):
        q = video_feat
        k = audio_feat
        v = audio_feat
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        x = self.norm(q + attn_output)
        return torch.mean(x, dim=1)

class MBT_Fusion(nn.Module):
    def __init__(self, embed_dim=512, n_head=1, n_layers=4, dropout=0.1, ffn_hidden=2048, num_bottleneck=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bottleneck = num_bottleneck
        self.bottleneck = nn.Parameter(torch.randn(num_bottleneck, embed_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,
                                       dim_feedforward=ffn_hidden, dropout=dropout,
                                       activation="relu", batch_first=True, norm_first=False)
            for _ in range(n_layers)
        ])

    def forward(self, audio_feat, video_feat):
        B, T_a, D = audio_feat.shape
        _, T_v, _ = video_feat.shape
        bottleneck = self.bottleneck.unsqueeze(0).expand(B, -1, -1)
        # Stage 1: Video + Bottleneck
        fused_video = torch.cat((video_feat, bottleneck), dim=1)
        for layer in self.layers:
            fused_video = layer(fused_video)
        video_fused = fused_video[:, :T_v, :]
        btlk_fused = fused_video[:, T_v:, :]
        # Stage 2: Audio + Updated Bottleneck
        fused_audio = torch.cat((audio_feat, btlk_fused), dim=1)
        for layer in self.layers:
            fused_audio = layer(fused_audio)
        audio_fused = fused_audio[:, :T_a, :]
        # Final fusion
        fused_embedding = torch.cat((audio_fused, video_fused), dim=1)
        return torch.mean(fused_embedding, dim=1)