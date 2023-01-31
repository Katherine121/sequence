import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Extractor(nn.Module):

    def __init__(self, backbone, dim):
        super(Extractor, self).__init__()
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, backbone,
                 num_classes1, num_classes2,
                 dim, depth, heads, mlp_dim, pool, len,
                 dim_head, dropout=0., emb_dropout=0.):
        super().__init__()
        self.extractor = Extractor(backbone, dim)
        self.extractor_dim = 576
        self.len = len

        # 576+2
        self.img_linear = nn.Linear(578, dim)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.len, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head_target = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.mlp_head_angle = nn.Sequential(
            # cbapd
            nn.LayerNorm(dim),
            nn.Hardtanh(),
            nn.Linear(dim, dim // 2),
            nn.Hardtanh(),
            nn.Linear(dim // 2, num_classes2)
        )

    def forward(self, img, ang):
        # 试试使用HOG特征
        # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
        img = self.extractor(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.extractor_dim)

        # # 不要终点的偏转方向
        # for i in range(1, self.len):
        #     ang[:, i, :] += ang[:, i - 1, :]
        # ang[:, 0, :] = 0
        # 5不变，4=3,3=2,2=1,1=0，0归为0

        ang[:, self.len - 2: self.len, :] = 0
        for i in range(1, self.len):
            ang[:, i, :] += ang[:, i - 1, :]

        # b,len,576->b,len,578
        img = torch.cat((img, ang), dim=-1)
        # b,len,578->b,len,512
        img = self.img_linear(img)

        b, n, _ = img.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        img = torch.cat((cls_tokens, img), dim=1)
        img[:, :1 + n, :] += self.pos_embedding[:, :1 + n, :]

        img = self.dropout(img)

        img = self.transformer(img)

        img = img.mean(dim=1) if self.pool == 'mean' else img[:, 0]

        return self.mlp_head_target(img), self.mlp_head_angle(img)
