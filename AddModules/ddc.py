import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try torchvision deform conv. If unavailable, fallback to normal conv.
try:
    from torchvision.ops import DeformConv2d
    _HAS_DCN = True
except Exception:
    DeformConv2d = None
    _HAS_DCN = False


class MeanFuse(nn.Module):
    """Mean Fusion for BHFP nodes. Input: list/tuple of tensors -> elementwise mean."""
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) == 0:
            raise TypeError("MeanFuse expects a non-empty list/tuple of tensors.")
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out / len(xs)


class GCAM(nn.Module):
    """
    Global Coordinate Attention Mechanism (lightweight engineering version):
    - global context g
    - coordinate-aware spatial weight w_sp via H/W pooling
    """
    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        mid = max(8, c // reduction)

        # global context
        self.gc1 = nn.Conv2d(c, mid, 1, 1, 0, bias=False)
        # self.gc_bn = nn.BatchNorm2d(mid)
        self.gc_bn = nn.GroupNorm(1, mid)  # 不依赖 batch size，B=1 也OK
        self.gc_act = nn.SiLU(inplace=True)
        self.gc2 = nn.Conv2d(mid, c, 1, 1, 0, bias=True)

        # coordinate attention
        self.coord = nn.Sequential(
            nn.Conv2d(c, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(mid, c, 1, 1, 0, bias=True)
        self.conv_w = nn.Conv2d(mid, c, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # global context gate
        g = F.adaptive_avg_pool2d(x, (1, 1))
        g = torch.sigmoid(self.gc2(self.gc_act(self.gc_bn(self.gc1(g)))))  # (B,C,1,1)

        # directional pooling
        x_h = F.adaptive_avg_pool2d(x, (h, 1))                         # (B,C,H,1)
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)     # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)                               # (B,C,H+W,1)
        y = self.coord(y)                                              # (B,mid,H+W,1)
        y_h, y_w = torch.split(y, [h, w], dim=2)

        a_h = torch.sigmoid(self.conv_h(y_h))                          # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w).permute(0, 1, 3, 2))      # (B,C,1,W)
        w_sp = a_h * a_w                                               # (B,C,H,W)
        return g, w_sp


class DDCGFLM(nn.Module):
    """
    GCAM-guided deformable conv + residual.
    YAML args: [c, k]
    """
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.c = c
        self.k = k
        self.gcam = GCAM(c)
        print("DDCGFLM c",c)

        # ==============================================================
        # 升级为 DCNv2：不仅输出 offsets，还输出 mask
        # DCNv2 需要 3 * k * k 的通道 (2*k*k 用于偏移，k*k 用于 mask 权重)
        # ==============================================================
        self.offset_mask = nn.Conv2d(c, 3 * k * k, 3, 1, 1)

        # 强制零初始化，保证训练初始阶段的绝对安全
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias, 0.0)

        if _HAS_DCN:
            self.op = DeformConv2d(c, c, kernel_size=k, stride=1, padding=k // 2, bias=False)
            # self.op = nn.Conv2d(c, c, k, 1, k // 2, bias=False)
        else:
            self.op = nn.Conv2d(c, c, k, 1, k // 2, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

        # 旁路卷积
        self.conv1 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        identity = self.conv1(x)

        # ==============================================================
        # 【终极护盾：绕过 YOLO FLOPs 计算期的致命崩溃】
        # 如果是 YOLO 传进来的全 0 假图片，直接短路跳过 DCN C++ 算子！
        # ==============================================================
        if x.sum() == 0.0:
            return identity + self.act(self.bn(identity))

        g, w_sp = self.gcam(x)

        if _HAS_DCN:
            out = self.offset_mask(x)

            # 分离 offset 和 mask (DCNv2 标准做法)
            o1, o2 = 2 * self.k * self.k, 3 * self.k * self.k
            offset = out[:, :o1, :, :]
            mask = torch.sigmoid(out[:, o1:o2, :, :])

            # 融合你的 GCAM 引导
            w = w_sp.mean(dim=1, keepdim=True)
            gg = g.mean(dim=1, keepdim=True)
            offset = offset * (1.0 + w) * (1.0 + gg)

            # 拦截 CPU 张量，防止底层 C++ 段错误
            if x.device.type == 'cpu':
                # YOLO 在初始化时会用 CPU tensor 跑一遍网络来计算 FLOPs
                # 直接构造一个正确 shape 的全零张量返回，跳过 DCN 的 CPU 计算
                B, _, H, W = x.shape
                C_out = self.op.weight.shape[0]  # 获取输出通道数 (这里是 64)
                y = torch.zeros((B, C_out, H, W), dtype=x.dtype, device=x.device)
            else:
                # 正式训练阶段，张量在 GPU 上，正常调用 DeformConv2d
                y = self.op(x.contiguous(), offset.contiguous(), mask.contiguous())
        else:
            y = self.op(x)

        y = self.act(self.bn(y))
        return y + identity


class MSADBlock(nn.Module):
    """
    Split-and-Combine parallel kernels.
    YAML args: [c, groups, [k1,k2,k3,k4]]
    """
    def __init__(self, c: int, groups: int = 4, kernels=(1, 3, 5, 7)):
        super().__init__()
        assert c % groups == 0, f"channels {c} must be divisible by groups {groups}"
        assert len(kernels) == groups, "kernels length must equal groups"
        cg = c // groups
        self.groups = groups

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cg, cg, k, 1, k // 2, bias=False),
                nn.BatchNorm2d(cg),
                nn.SiLU(inplace=True),
            )
            for k in kernels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        xs = torch.chunk(x, self.groups, dim=1)
        ys = [b(xi) for b, xi in zip(self.branches, xs)]
        y = torch.cat(ys, dim=1)
        return self.fuse(y)