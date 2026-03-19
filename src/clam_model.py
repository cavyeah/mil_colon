import torch
import torch.nn as nn


class AttentionNet(nn.Module):

    def __init__(self, in_dim=2048, hidden_dim=512, topk_ratio=0.1):
        super().__init__()

        self.topk_ratio = topk_ratio

        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):

        A = self.attention(x)

        A = A - torch.max(A)

        A = torch.softmax(A, dim=0)

        num_patches = x.shape[0]

        k = max(1, int(num_patches * self.topk_ratio))

        top_idx = torch.topk(A.squeeze(), k).indices

        x = x[top_idx]

        A = A[top_idx]

        A = A / torch.sum(A)

        M = torch.sum(A * x, dim=0)

        return M, A


class CLAM_SB(nn.Module):

    def __init__(self, in_dim=2048):
        super().__init__()

        self.attention_net = AttentionNet(in_dim)

        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x):

        M, A = self.attention_net(x)

        logits = self.classifier(M)

        return logits, A