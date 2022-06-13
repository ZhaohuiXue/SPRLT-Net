# -*- coding: utf-8 -*-
"""
@author:
"""
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, in_chnls, ratio=16):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        x = rearrange(x, 'b w h c->b c w h')
        y = self.squeeze(x)
        y = self.compress(y)
        y = F.relu(y)
        y = self.excitation(y)
        x = x * y
        x = rearrange(x, 'b c w h->b w h c')
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class Residualforword(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class LinearProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        x = self.linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.se=SE(dim)

    def forward(self, x):
        return self.net(x)+x


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, layer, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        # self.topatch = nn.Unfold(kernel_size=window_size, stride=3)
        # self.backpatch = nn.Fold(output_size=(9, 9), kernel_size=(window_size, window_size), stride=(3, 3))
        self.topatch = nn.Unfold(kernel_size=window_size, stride=2)
        self.backpatch = nn.Fold(output_size=(9, 9), kernel_size=(window_size, window_size), stride=(2, 2))

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(self.window_size) + self.window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * self.window_size - 1, 2 * self.window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(self.window_size ** 2, self.window_size ** 2))
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, int(dim / 2)),
            nn.LayerNorm(int(dim / 2)),
            nn.GELU(),
            nn.Linear(int(dim / 2), dim),
        )

    def forward(self, x):
        b, n_h, n_w, _, h = *x.shape, self.heads  # 符号*代表序列解包

        qkv = self.to_qkv(x)
        aa = qkv.permute(0, 3, 1, 2)
        aa = self.topatch(aa)
        aa = rearrange(aa, 'b (c nw nh) n ->b n (nw nh) c', nw=self.window_size, nh=self.window_size)
        aa = aa.chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, 'b n wh (h d) -> b h n wh d', h=h), aa)

        attn = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            attn += self.pos_embedding[
                self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(torch.long)]
        else:
            attn += self.pos_embedding

        attn = self.softmax(attn)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h n wh d -> b (wh h d) n')
        out = self.backpatch(out)

        out = self.to_out(out.permute(0, 2, 3, 1))

        return out


class Block(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, layer, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     layer=layer,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        # self.attention_block = Residual(PreNorm(dim,Pooling(pool_size=window_size)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class SPRLT(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=103, num_classes=9, head_dim=32, window_size=3,
                 relative_pos_embedding=True):
        super().__init__()

        self.patch_partition = LinearProject(in_channels=channels, out_channels=hidden_dim)
        self.layers = nn.ModuleList([])

        for ii in range(layers):
            self.layers.append(Block(dim=hidden_dim, heads=heads, head_dim=head_dim, mlp_dim=hidden_dim * 4,
                                     shifted=True, layer=ii, window_size=window_size,
                                     relative_pos_embedding=relative_pos_embedding)
                               )
        self.posision = nn.Parameter(torch.rand(9, 9))

        self.mask = 1 /layers/ torch.tensor([[1, 1, 2, 2, 3, 2, 2, 1, 1],
                                      [1, 1, 2, 2, 3, 2, 2, 1, 1],
                                      [2, 2, 4, 4, 6, 4, 4, 2, 2],
                                      [2, 2, 4, 4, 6, 4, 4, 2, 2],
                                      [3, 3, 6, 6, 9, 6, 6, 3, 3],
                                      [2, 2, 4, 4, 6, 4, 4, 2, 2],
                                      [2, 2, 4, 4, 6, 4, 4, 2, 2],
                                      [1, 1, 2, 2, 3, 2, 2, 1, 1],
                                      [1, 1, 2, 2, 3, 2, 2, 1, 1]]).cuda()

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 4)),
            nn.LayerNorm(int(hidden_dim / 4)),
            nn.GELU(),
            nn.Linear(int(hidden_dim / 4), num_classes))

    def forward(self, x):  # x: [B, W, H, C]        )

        x = self.patch_partition(x)
        # x += self.pos_embedding
        # x = self.se(x)
        for block in self.layers:
            x = block(x)

        x = x.permute(0, 3, 1, 2)
        # x = x * self.mask
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


if __name__ == "__main__":
    num_PC = 103
    classnum = 9
    net = SPRLT(
        hidden_dim=128,  # 96
        layers=(3),
        heads=(12),  ##more
        channels=num_PC,
        num_classes=classnum,
        head_dim=24,
        window_size=5,
        relative_pos_embedding=True
    ).cuda()

    img = torch.rand(64, 9, 9, num_PC).cuda()
    # summary(net, (9, 9, 103))
    res = net(img)
    # trace_model=torch.jit.trace(net,img)
    # trace_model.save('trace.pt')
    print(res.shape)
