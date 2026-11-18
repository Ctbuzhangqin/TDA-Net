import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from transformers.activations import ACT2FN
from .utils import window_partition_2d, window_reverse_2d, get_window_size


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125
    ):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size), padding=0, groups=gc
        )
        self.pad_w = band_kernel_size // 2
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def _conv_hw(self, x_hw):
        return self.dwconv_hw(x_hw)

    def forward(self, x):
        (x_id, x_hw, x_w, x_h) = torch.split(x, self.split_indexes, dim=1)
        x_w = F.pad(x_w, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        return torch.cat(
            (x_id, self._conv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class Mlp(nn.Module):
    def __init__(self, config, in_dims, out_dims=None):
        super().__init__()
        hidden_dims = int(in_dims * config.mlp_ratio)
        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.act = ACT2FN[config.hidden_act]
        if out_dims is None:
            out_dims = in_dims
        self.fc2 = nn.Linear(hidden_dims, out_dims)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseDispatcher(object):
    """稀疏路由分发器：将 token 按 gate 分配给对应 expert，再加权合并输出。"""

    def __init__(self, num_experts, gates):
        """
        Args:
            num_experts: int
            gates: (N, num_experts)，N = B*H*W
        """
        self._gates = gates
        self._num_experts = num_experts
        (sorted_experts, index_sorted_experts) = torch.nonzero(gates).sort(0)
        (_, self._expert_index) = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out):
        stitched = torch.cat(expert_out, 0)
        stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SpMoeGate(nn.Module):
    """lead_time 条件化的 Top-K 门控。

    用一个大 Linear 输出 max_t * num_experts 维，reshape 后按 lead_t 索引取对应的 gate scores。
    全部 max_t 组的 scores 均通过 sigmoid 后返回，用于计算 aux loss。
    """

    def __init__(self, dim, topk, num_experts, max_t):
        super().__init__()
        self.dim = dim
        self.k = topk
        self.num_experts = num_experts
        self.max_t = max_t
        self.w_gate = nn.Linear(dim, max_t * num_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, lead_t):
        """
        Args:
            x: (B, H, W, D)
            lead_t: (B,) 值域 [0, max_t-1]
        Returns:
            gates: (B*H*W, num_experts) 稀疏 gate（仅 top-k 非零）
            all_scores: (max_t, num_experts) 各 lead_t 的平均 gate score，用于 aux loss
        """
        (B, H, W, D) = x.shape
        x_flat = x.reshape(B * H * W, D)
        raw_scores = torch.sigmoid(self.w_gate(x_flat))
        raw_scores = raw_scores.reshape(B * H * W, self.max_t, self.num_experts)
        lead_t_expanded = (
            lead_t.unsqueeze(1).unsqueeze(1).expand(B, H, W).reshape(B * H * W)
        )
        lead_t_idx = (
            lead_t_expanded.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_experts)
        )
        selected_scores = raw_scores.gather(1, lead_t_idx).squeeze(1)
        (top_k_values, top_k_indices) = selected_scores.topk(
            min(self.k, self.num_experts), dim=1
        )
        top_k_gates = self.softmax(top_k_values)
        zeros = torch.zeros_like(selected_scores, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates.to(zeros.dtype))
        all_scores = []
        for t in range(self.max_t):
            mask = lead_t_expanded == t
            if mask.any():
                all_scores.append(raw_scores[mask].mean(dim=0))
            else:
                all_scores.append(raw_scores.mean(dim=0))
        all_scores = torch.stack(all_scores, dim=0)
        all_scores_diag = torch.stack(
            [all_scores[t, t] for t in range(self.max_t)], dim=0
        )
        return (gates, all_scores_diag)


class SpMoE(nn.Module):
    """Shared-Private MoE：shared expert 永远激活，routed experts Top-K 激活。

    Args:
        config: 模型配置
        in_dims: 输入维度
        out_dims: 输出维度（默认等于 in_dims）
        num_routed_experts: routed expert 数量（默认 9）
        topk: 每次激活的 routed expert 数量（默认 2）
    """

    def __init__(self, config, in_dims, out_dims=None, num_routed_experts=9, topk=2):
        super().__init__()
        if out_dims is None:
            out_dims = in_dims
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_routed_experts = num_routed_experts
        self.topk = topk
        self.max_t = config.max_t
        hidden_dims = int(in_dims * config.mlp_ratio)
        self.shared_expert = Mlp(config, in_dims, out_dims)
        self.routed_experts = nn.ModuleList(
            [Mlp(config, in_dims, out_dims) for _ in range(num_routed_experts)]
        )
        self.gate = SpMoeGate(
            dim=in_dims, topk=topk, num_experts=num_routed_experts, max_t=config.max_t
        )
        self.aux_loss_scores = None

    def forward(self, x, lead_t):
        """
        Args:
            x: (B, H, W, C)
            lead_t: (B,) 值域 [0, max_t-1]
        Returns:
            out: (B, H, W, out_dims)
        """
        (B, H, W, C) = x.shape
        lead_t = lead_t % self.max_t
        (gates, all_scores) = self.gate(x, lead_t)
        self.aux_loss_scores = all_scores
        x_flat = x.reshape(B * H * W, C)
        z = self.shared_expert(x.reshape(B * H * W, 1, C).reshape(B, H, W, C))
        z = z.reshape(B * H * W, self.out_dims)
        dispatcher = SparseDispatcher(self.num_routed_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = []
        for i in range(self.num_routed_experts):
            if expert_inputs[i].shape[0] == 0:
                expert_outputs.append(expert_inputs[i].new_zeros(0, self.out_dims))
            else:
                inp = expert_inputs[i].unsqueeze(1).unsqueeze(1)
                out = self.routed_experts[i](inp)
                expert_outputs.append(out.squeeze(1).squeeze(1))
        y = dispatcher.combine(expert_outputs)
        return (y + z).reshape(B, H, W, self.out_dims)


def modulate(x, shift, scale):
    """adaLN 调制：x * (1 + scale) + shift
    Args:
        x: (B, H, W, C)
        shift, scale: (B, C)
    """
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)


class MonthEmbedder(nn.Module):
    """月份+lead_time 联合编码器。
    month(0-11): 周期性 → sin/cos(2π·month/12)，2D
    lead_time(0-23): 非周期性 → sin/cos(lead/max_lead·π)，2D
    拼接后 4D → Linear → mon_hidden_dim
    """

    def __init__(self, mon_hidden_dim, max_lead=24):
        super().__init__()
        self.max_lead = max_lead
        self.proj = nn.Linear(4, mon_hidden_dim)

    def forward(self, month, lead_time):
        """
        Args:
            month: (B,) tensor, 值域 [0, 11]
            lead_time: (B,) tensor, 值域 [0, max_lead-1]
        Returns:
            (B, mon_hidden_dim) tensor
        """
        month_rad = (month.float() - 1.0) * (2.0 * 3.14159265359 / 12.0)
        month_enc = torch.stack([torch.sin(month_rad), torch.cos(month_rad)], dim=-1)
        lead_rad = lead_time.float() / self.max_lead * 3.14159265359
        lead_enc = torch.stack([torch.sin(lead_rad), torch.cos(lead_rad)], dim=-1)
        enc = torch.cat([month_enc, lead_enc], dim=-1)
        return self.proj(enc)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        input_shape,
        patch_size,
        in_chans,
        embed_dim,
        patch_norm,
        ape,
        mask_token=False,
        lonlat_emb=False,
        lat_space=None,
        lon_space=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.ape = ape
        self.lonlat_emb = lonlat_emb
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if patch_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None
        if self.ape:
            Hp = int(np.ceil(input_shape[0] / patch_size[0]))
            Wp = int(np.ceil(input_shape[1] / patch_size[1]))
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, Hp * Wp, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward(self, x, mask=None):
        (_, _, H, W) = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        (Wh, Ww) = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        (B, L, _) = x.shape
        if mask is not None:
            raise ValueError(
                "Masked pretraining is not part of the released TDA-Net model."
            )
        if self.norm is not None:
            x = self.norm(x)
        if self.ape:
            pos_embed = self.absolute_pos_embed
            x = x + pos_embed
        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        (B, H, W, C) = x.shape
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        x = self.expand(x)
        (B, H, W, C) = x.shape
        x = rearrange(
            x, "b h w (p2 p1 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4
        ).contiguous()
        x = self.norm(x)
        return x


class RotaryTimeEmbed(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        theta = 10000 ** (torch.arange(dim).div(2, rounding_mode="floor") * 2 / dim)
        self.register_buffer("theta", theta)

    def forward(self, x, time):
        theta = torch.einsum("i,j->ij", time, self.theta)
        cos_enc = torch.cos(theta)[:, :, None, None]
        sin_enc = torch.sin(theta)[:, :, None, None]
        x_rot = torch.stack([-x[:, 1::2], x[:, ::2]], dim=1).reshape_as(x)
        x = x * cos_enc + x_rot * sin_enc
        return x


class RotaryPosEmbed2D(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords = torch.stack(
            torch.meshgrid([coords_0, coords_1], indexing="ij")
        ).reshape(2, -1)
        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 10000 ** (-freq_seq1)
        inv_freq2 = 10000 ** (-freq_seq2)
        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2
        self.register_buffer(
            "sin1", torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        )
        self.register_buffer(
            "cos1", torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        )
        self.register_buffer(
            "sin2", torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        )
        self.register_buffer(
            "cos2", torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        )

    def forward(self, x):
        (x11, x21, x12, x22) = x.split(
            [self.dim1_size, self.dim2_size, self.dim1_size, self.dim2_size], dim=-1
        )
        res = torch.cat(
            [
                x11 * self.cos1 - x12 * self.sin1,
                x21 * self.cos2 - x22 * self.sin2,
                x12 * self.cos1 + x11 * self.sin1,
                x22 * self.cos2 + x21 * self.sin2,
            ],
            dim=-1,
        )
        return res


class WindowAttention(nn.Module):
    def __init__(self, config, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = config.qk_scale or head_dim ** (-0.5)
        self.head_dim = head_dim
        self.pos_embed = RotaryPosEmbed2D(window_size, head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        (B_, N, C) = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        q = self.pos_embed(
            q.reshape(-1, *self.window_size, C // self.num_heads)
        ).reshape(B_, self.num_heads, -1, C // self.num_heads)
        k = self.pos_embed(
            k.reshape(-1, *self.window_size, C // self.num_heads)
        ).reshape(B_, self.num_heads, -1, C // self.num_heads)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class InceptionSwinLayer(nn.Module):
    """Inception token mixer with monthly modulation and optional shared-private experts."""

    def __init__(
        self,
        config,
        window_size,
        dim,
        num_heads,
        shift_size=(0, 0),
        drop_path=0.0,
        is_moe=False,
    ):
        super().__init__()
        self.dim = dim
        band_kernel_size = getattr(config, "inception_band_kernel_size", 11)
        branch_ratio = getattr(config, "inception_branch_ratio", 0.125)
        self.use_lat_gate = getattr(config, "use_lat_gate", False)
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.token_mixer = InceptionDWConv2d(
            in_channels=dim,
            square_kernel_size=3,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.is_moe = is_moe
        self.sp_moe = getattr(config, "sp_moe", False)
        if is_moe:
            self.mlp = SpMoE(config, in_dims=dim)
        else:
            self.mlp = Mlp(config, in_dims=dim)
        self.use_month_emb = getattr(config, "use_month_emb", False)
        mon_hidden_dim = getattr(config, "mon_hidden_dim", 512)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(mon_hidden_dim, 6 * dim)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        self.gradient_checkpointing = False

    def forward_part1(self, x, shift=None, scale=None):
        (B, H, W, C) = x.shape
        if shift is not None:
            x = modulate(self.norm1(x), shift, scale)
        else:
            x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.token_mixer(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_part2(self, x, lead_t, shift=None, scale=None):
        if shift is not None:
            normed = modulate(self.norm2(x), shift, scale)
        else:
            normed = self.norm2(x)
        if self.is_moe:
            return self.drop_path(self.mlp(normed, lead_t))
        else:
            return self.drop_path(self.mlp(normed))

    def forward(self, x, mask_matrix, mask_matrix_shifted, lead_t=None, mon_emb=None):
        if mon_emb is not None:
            mod = self.adaLN_modulation(mon_emb)
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                mod.chunk(6, dim=-1)
            )
            gate_msa = gate_msa.unsqueeze(1).unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1).unsqueeze(1)
            shortcut = x
            x = self.forward_part1(x, shift_msa, scale_msa)
            x = shortcut + gate_msa * self.drop_path(x)
            x = x + gate_mlp * self.forward_part2(x, lead_t, shift_mlp, scale_mlp)
        else:
            shortcut = x
            if self.gradient_checkpointing:
                x = checkpoint.checkpoint(self.forward_part1, x)
            else:
                x = self.forward_part1(x)
            x = shortcut + self.drop_path(x)
            if self.gradient_checkpointing:
                x = x + checkpoint.checkpoint(self.forward_part2, x, lead_t)
            else:
                x = x + self.forward_part2(x, lead_t)
        return x


class SwinLayer(nn.Module):
    def __init__(
        self,
        config,
        window_size,
        dim,
        num_heads,
        shift_size=(0, 0),
        drop_path=0.0,
        is_moe=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = config.mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], (
            "shift_size must in 0-window_size"
        )
        assert 0 <= self.shift_size[1] < self.window_size[1], (
            "shift_size must in 0-window_size"
        )
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = WindowAttention(
            config, dim, window_size=self.window_size, num_heads=num_heads
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.is_moe = is_moe
        self.sp_moe = getattr(config, "sp_moe", False)
        if is_moe:
            self.mlp = SpMoE(config, in_dims=dim)
        else:
            self.mlp = Mlp(config, in_dims=dim)
        self.use_month_emb = getattr(config, "use_month_emb", False)
        mon_hidden_dim = getattr(config, "mon_hidden_dim", 512)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(mon_hidden_dim, 6 * dim)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        self.gradient_checkpointing = False

    def forward_part1(
        self, x, mask_matrix, mask_matrix_shifted, shift=None, scale=None
    ):
        (B, H, W, C) = x.shape
        (window_size, shift_size) = get_window_size(
            (H, W), self.window_size, self.shift_size
        )
        if shift is not None:
            x = modulate(self.norm1(x), shift, scale)
        else:
            x = self.norm1(x)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        (_, Hp, Wp, _) = x.shape
        if any((i > 0 for i in shift_size)):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
            )
            attn_mask = mask_matrix_shifted
        else:
            shifted_x = x
            attn_mask = mask_matrix
        x_windows = window_partition_2d(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *window_size + (C,))
        shifted_x = window_reverse_2d(attn_windows, window_size, B, Hp, Wp)
        if any((i > 0 for i in shift_size)):
            x = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x, lead_t, shift=None, scale=None):
        if shift is not None:
            normed = modulate(self.norm2(x), shift, scale)
        else:
            normed = self.norm2(x)
        if self.is_moe:
            return self.drop_path(self.mlp(normed, lead_t))
        else:
            return self.drop_path(self.mlp(normed))

    def forward(self, x, mask_matrix, mask_matrix_shifted, lead_t=None, mon_emb=None):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
            mon_emb: (B, mon_hidden_dim) month embedding, optional.
        """
        if mon_emb is not None:
            mod = self.adaLN_modulation(mon_emb)
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                mod.chunk(6, dim=-1)
            )
            gate_msa = gate_msa.unsqueeze(1).unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1).unsqueeze(1)
            shortcut = x
            x = self.forward_part1(
                x, mask_matrix, mask_matrix_shifted, shift_msa, scale_msa
            )
            x = shortcut + gate_msa * self.drop_path(x)
            x = x + gate_mlp * self.forward_part2(x, lead_t, shift_mlp, scale_mlp)
        else:
            shortcut = x
            if self.gradient_checkpointing:
                x = checkpoint.checkpoint(
                    self.forward_part1, x, mask_matrix, mask_matrix_shifted
                )
            else:
                x = self.forward_part1(x, mask_matrix, mask_matrix_shifted)
            x = shortcut + self.drop_path(x)
            if self.gradient_checkpointing:
                x = x + checkpoint.checkpoint(self.forward_part2, x, lead_t)
            else:
                x = x + self.forward_part2(x, lead_t)
        return x


class SwinEncoderStage(nn.Module):
    def __init__(
        self,
        config,
        window_size,
        dim,
        depth,
        num_heads,
        drop_path=0.0,
        downsample=None,
        is_atmo=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple((i // 2 for i in window_size))
        self.depth = depth
        self.is_moe = getattr(config, "is_moe", False)
        if is_atmo:
            self.is_moe_encoder = getattr(config, "is_moe_atmo", True)
        else:
            self.is_moe_encoder = getattr(config, "is_moe_encoder", True)
        use_inception = getattr(config, "use_inception", False)
        LayerClass = InceptionSwinLayer if dim != config.lg_hidden_dim else SwinLayer
        self.blocks = nn.ModuleList(
            [
                LayerClass(
                    config,
                    window_size,
                    dim=dim,
                    num_heads=num_heads,
                    shift_size=(0, 0) if i % 2 == 0 else self.shift_size,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    is_moe=self.is_moe and self.is_moe_encoder,
                )
                for i in range(depth)
            ]
        )
        self.use_inception = use_inception
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim)

    def forward(
        self,
        x,
        land_mask_pad=None,
        land_mask_pad_shifted=None,
        lead_t=None,
        mon_emb=None,
    ):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
            mon_emb: (B, mon_hidden_dim) month embedding, optional.
        """
        (B, H, W, C) = x.shape
        attn_mask = None
        attn_mask_shifted = None
        for blk in self.blocks:
            if self.is_moe:
                x = blk(x, attn_mask, attn_mask_shifted, lead_t, mon_emb=mon_emb)
            else:
                x = blk(x, attn_mask, attn_mask_shifted, mon_emb=mon_emb)
        x = x.view(B, H, W, -1)
        x_before_downsample = x
        if self.downsample is not None:
            x = self.downsample(x)
        return (x, x_before_downsample)


class SwinDecoderStage(nn.Module):
    def __init__(
        self, config, window_size, dim, depth, num_heads, drop_path=0.0, upsample=None
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple((i // 2 for i in window_size))
        self.depth = depth
        self.is_moe = getattr(config, "is_moe", False)
        self.is_moe_decoder = getattr(config, "is_moe_decoder", True)
        use_inception = getattr(config, "use_inception", False)
        LayerClass = InceptionSwinLayer
        self.blocks = nn.ModuleList(
            [
                LayerClass(
                    config,
                    window_size,
                    dim=dim,
                    num_heads=num_heads,
                    shift_size=(0, 0) if i % 2 == 0 else self.shift_size,
                    drop_path=drop_path[depth - 1 - i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    is_moe=self.is_moe and self.is_moe_decoder,
                )
                for i in range(depth)
            ]
        )
        self.use_inception = use_inception
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim)

    def forward(
        self,
        x,
        land_mask_pad=None,
        land_mask_pad_shifted=None,
        lead_t=None,
        mon_emb=None,
    ):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
            mon_emb: (B, mon_hidden_dim) month embedding, optional.
        """
        (B, H, W, C) = x.shape
        attn_mask = None
        attn_mask_shifted = None
        for blk in self.blocks:
            if self.is_moe:
                x = blk(x, attn_mask, attn_mask_shifted, lead_t, mon_emb=mon_emb)
            else:
                x = blk(x, attn_mask, attn_mask_shifted, mon_emb=mon_emb)
        x = x.view(B, H, W, -1)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
