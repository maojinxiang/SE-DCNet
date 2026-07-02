import torch
import torch.nn as nn
import torchvision.models as models

# ===========================
# 核心改进：SE 注意力模块
# ===========================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Channel)
        b, c = x.size()
        # 变成 (Batch, Channel, 1) 以便进行 Channel-wise 操作
        y = x.view(b, c, 1)
        # Squeeze: 全局平均池化 (其实这里输入已经是特征向量了，这步主要是为了形式统一)
        y = self.avg_pool(y).view(b, c)
        # Excitation: 全连接层计算权重
        y = self.fc(y).view(b, c, 1)
        # Scale: 原始特征 * 权重
        return x * y.view(b, c)

# ===========================
# 双通道模型 (带注意力)
# ===========================
class DualChannelDroneNet(nn.Module):
    def __init__(self, num_classes=9):
        super(DualChannelDroneNet, self).__init__()
        
        # --- 分支 1: 1D-CNN ---
        self.branch_1d = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(256, 512, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # --- 分支 2: EfficientNet-B0 ---
        self.effnet = models.efficientnet_b0(weights='DEFAULT')
        original_conv = self.effnet.features[0][0]
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels=2, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        self.branch_2d = nn.Sequential(
            self.effnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # --- 融合部分 ---
        # 1D 输出: 512
        # 2D 输出: 1280
        fusion_dim = 512 + 1280 
        
        # 【关键改进】加入 SE 注意力
        self.attention = SEBlock(channel=fusion_dim, reduction=16)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, seq):
        # 1D 特征
        x_seq = self.branch_1d(seq)
        x_seq = x_seq.view(x_seq.size(0), -1) # (B, 512)
        
        # 2D 特征
        x_img = self.branch_2d(img) # (B, 1280)
        
        # 拼接
        combined = torch.cat((x_img, x_seq), dim=1) # (B, 1792)
        
        # 【关键改进】注意力加权
        # 如果 1D 特征质量差，SEBlock 会自动给后 512 维很低的权重
        combined = self.attention(combined)
        
        out = self.classifier(combined)
        return out

# ===========================
# 单通道模型 (保持不变，用于对比)
# ===========================
class DroneNet_1D_Only(nn.Module):
    def __init__(self, num_classes=9):
        super(DroneNet_1D_Only, self).__init__()
        self.branch_1d = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(256, 512, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, seq):
        x = self.branch_1d(seq)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class DroneNet_2D_Only(nn.Module):
    def __init__(self, num_classes=9):
        super(DroneNet_2D_Only, self).__init__()
        self.effnet = models.efficientnet_b0(weights='DEFAULT')
        original_conv = self.effnet.features[0][0]
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels=2, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        self.branch_2d = nn.Sequential(
            self.effnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, seq):
        x = self.branch_2d(img)
        out = self.classifier(x)
        return out


class DroneNet_ResNet_Only(nn.Module):
    def __init__(self, num_classes=9):
        super(DroneNet_ResNet_Only, self).__init__()
        # 兼容不同 torchvision 版本：新版本用 weights，旧版本用 pretrained。
        try:
            self.backbone = models.resnet18(weights=None)
        except TypeError:
            self.backbone = models.resnet18(pretrained=False)

        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, img, seq):
        return self.backbone(img)


class DroneNet_LSTM_Only(nn.Module):
    def __init__(self, num_classes=9):
        super(DroneNet_LSTM_Only, self).__init__()
        self.seq_pool = nn.AdaptiveAvgPool1d(2048)
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, seq):
        # 先压缩时间维，避免直接处理百万级序列导致训练不可行。
        seq_pooled = self.seq_pool(seq)
        seq_tokens = seq_pooled.transpose(1, 2)
        outputs, _ = self.lstm(seq_tokens)
        return self.classifier(outputs[:, -1, :])


class TemporalDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(TemporalDilatedBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class DroneNet_TCN_Only(nn.Module):
    """
    纯时域 TCN 对照模型：和 1D-CNN 做同域对比。
    """
    def __init__(self, num_classes=9):
        super(DroneNet_TCN_Only, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        self.block1 = TemporalDilatedBlock(64, 128, dilation=1)
        self.down1 = nn.MaxPool1d(2)

        self.block2 = TemporalDilatedBlock(128, 256, dilation=2)
        self.down2 = nn.MaxPool1d(2)

        self.block3 = TemporalDilatedBlock(256, 512, dilation=4)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, img, seq):
        x = self.stem(seq)
        x = self.down1(self.block1(x))
        x = self.down2(self.block2(x))
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)
    
# ===========================
# CNN-LSTM baseline
# ===========================
class DroneNet_CNN_LSTM(nn.Module):
    """
    CNN-LSTM baseline for RF fingerprinting.
    The 1D CNN first downsamples the long I/Q sequence, and the BiLSTM then models temporal dependencies.
    """
    def __init__(self, num_classes=9, lstm_hidden=128, lstm_layers=2, token_len=512):
        super(DroneNet_CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(token_len)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=0.2 if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, seq):
        # seq: (B, 2, L)
        x = self.cnn(seq)              # (B, 256, token_len)
        x = x.transpose(1, 2)          # (B, token_len, 256)
        outputs, _ = self.lstm(x)      # (B, token_len, 2*lstm_hidden)
        feat = outputs[:, -1, :]       # use last token
        return self.classifier(feat)

# ===========================
# Cross-Attention baseline
# ===========================
class DroneNet_CrossAttention(nn.Module):
    """
    Dual-domain cross-attention baseline.
    It uses the same 1D and 2D feature extractors as SE-DCNet, but replaces SE fusion with attention-based fusion.
    """
    def __init__(self, num_classes=9, embed_dim=256, num_heads=4):
        super(DroneNet_CrossAttention, self).__init__()

        # --- 1D branch ---
        self.branch_1d = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(256, 512, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # --- 2D branch ---
        self.effnet = models.efficientnet_b0(weights='DEFAULT')
        original_conv = self.effnet.features[0][0]
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        self.branch_2d = nn.Sequential(
            self.effnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # project both domains to the same embedding dimension
        self.proj_1d = nn.Linear(512, embed_dim)
        self.proj_2d = nn.Linear(1280, embed_dim)

        # token-level cross-domain attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, seq):
        x_seq = self.branch_1d(seq).view(seq.size(0), -1)  # (B, 512)
        x_img = self.branch_2d(img)                        # (B, 1280)

        token_seq = self.proj_1d(x_seq)                    # (B, D)
        token_img = self.proj_2d(x_img)                    # (B, D)

        tokens = torch.stack([token_img, token_seq], dim=1) # (B, 2, D)

        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        fused = tokens.flatten(start_dim=1)                 # (B, 2D)
        return self.classifier(fused)
    

class DualChannelConcatNet(DualChannelDroneNet):
    """
    Direct Concatenation fusion baseline.
    Same two branches as SE-DCNet, but without SE attention.
    """
    def __init__(self, num_classes=9):
        super(DualChannelConcatNet, self).__init__(num_classes=num_classes)
        self.attention = nn.Identity()


class DualChannelWeightedFusionNet(DualChannelDroneNet):
    """
    Learnable weighted fusion baseline.
    Same two branches as SE-DCNet, but uses two learnable scalar weights
    for 2D and 1D features before concatenation.
    """
    def __init__(self, num_classes=9):
        super(DualChannelWeightedFusionNet, self).__init__(num_classes=num_classes)
        self.fusion_weight = nn.Parameter(torch.ones(2))

    def forward(self, img, seq):
        x_seq = self.branch_1d(seq)
        x_seq = x_seq.view(x_seq.size(0), -1)

        x_img = self.branch_2d(img)

        weights = torch.softmax(self.fusion_weight, dim=0)
        x_img = weights[0] * x_img
        x_seq = weights[1] * x_seq

        combined = torch.cat((x_img, x_seq), dim=1)
        out = self.classifier(combined)
        return out