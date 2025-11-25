# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -----------------------------
# ConvLSTM (single-layer)
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=pad, bias=bias)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, x):
        # x: (B, C, H, W)
        B, _, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        return h, c


# -----------------------------
# Deform-like Attention Module
# (lightweight: MultiheadAttention over flattened spatial tokens)
# -----------------------------
class DeformableFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        # registered projection layer for skip -> dim (created lazily if needed)
        self.skip_proj = None
        self.skip_proj_in_ch = None

    def forward(self, x, skip):
        B, C, H, W = x.shape
        if skip is None:
            return x

        # ensure spatial size
        if skip.shape[-2:] != (H, W):
            skip = F.interpolate(skip, size=(H, W), mode='bilinear', align_corners=False)

        # create/register a projection if skip channels != C
        if skip.shape[1] != C:
            if (self.skip_proj is None) or (self.skip_proj_in_ch != skip.shape[1]):
                # create a new projection and register it
                self.skip_proj = nn.Conv2d(skip.shape[1], C, kernel_size=1).to(x.device)
                self.skip_proj_in_ch = skip.shape[1]
            skip = self.skip_proj(skip)

        q = x.flatten(2).transpose(1, 2)   # (B, HW, C)
        k = skip.flatten(2).transpose(1, 2)
        v = k
        attn_out, _ = self.attn(q, k, v)   # (B, HW, C)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return self.proj(attn_out + x)


# -----------------------------
# Mix FFN (depthwise conv + 1x1 project)
# -----------------------------
class MixFFN(nn.Module):
    def __init__(self, dim, expansion=4, drop=0.0):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        # depthwise conv for local mixing
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(self.fc2(x))
        return x


# -----------------------------------------
# Decoder block: Up + Fusion + ConvLSTM + MixFFN
# -----------------------------------------
class DecoderBlockEnhanced(nn.Module):
    def __init__(self, up_in_ch, skip_ch, out_ch, sim_state_channels, use_deconv=True):
        """
        up_in_ch: channels of incoming (deeper) feature
        skip_ch: channels of skip (encoder) feature (0 or None if absent)
        out_ch: output channels of this block
        sim_state_channels: channels in each SIM state (2*N)
        """
        super().__init__()
        self.use_deconv = use_deconv
        if use_deconv:
            self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(up_in_ch, out_ch, kernel_size=1)
            )

        # Fusion module expects channels == out_ch
        self.fusion = DeformableFusion(dim=out_ch, num_heads=max(1, out_ch // 64))

        # project sim -> out_ch if sim channels exist
        self.sim_proj = nn.Conv2d(sim_state_channels, out_ch, kernel_size=1)

        # ConvLSTM expects input channels == out_ch
        self.clstm = ConvLSTMCell(input_dim=out_ch, hidden_dim=out_ch, kernel_size=3)

        self.mix_ffn = MixFFN(out_ch, expansion=4, drop=0.0)

    def forward(self, x, skip, sim_states_list):
        """
        x: (B, up_in_ch, H/2, W/2)
        skip: (B, skip_ch, H, W) or None
        sim_states_list: list of D tensors resized to this block's spatial size, each (B, sim_ch, H, W)
        """
        x = self.up(x)  # now (B, out_ch, H, W) if ConvTranspose out channels == out_ch
        # ensure spatial match
        if skip is not None and x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # fusion: if skip is None, fusion returns x
        x = self.fusion(x, skip)

        # ConvLSTM initial state (zeros)
        h, c = self.clstm.init_state(x)

        # run through D timesteps injecting SIM guidance
        for sim in sim_states_list:
            guided = x + self.sim_proj(sim)
            h, c = self.clstm(guided, (h, c))

        out = self.mix_ffn(h)
        return out


# ------------------------------------------------
# Adapt Swin patch embed to accept custom in_channels
# ------------------------------------------------
def _adapt_swin_input(model, in_ch):
    """
    Replaces patch_embed.proj to accept in_ch input channels.
    If the original proj had 3-channel pretrained weights and in_ch == 1,
    initialize new proj by averaging RGB weights -> preserves pretrained signal.
    """
    patch_embed = model.patch_embed
    old = patch_embed.proj

    # obtain layer attributes (safe fallback)
    kernel_size = old.kernel_size if hasattr(old, "kernel_size") else (4, 4)
    stride = old.stride if hasattr(old, "stride") else (4, 4)
    padding = old.padding if hasattr(old, "padding") else 0
    out_ch = old.out_channels if hasattr(old, "out_channels") else old.weight.shape[0]

    new_proj = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    # If old has pretrained weights with 3 input channels and we adapt to 1, copy mean(R,G,B)
    if hasattr(old, "weight"):
        old_w = old.weight.data
        if old_w.ndim == 4 and old_w.shape[1] == 3 and in_ch == 1:
            # mean across input channel dim to produce single-channel kernel
            new_proj.weight.data = old_w.mean(dim=1, keepdim=True).clone()
            if hasattr(old, "bias") and old.bias is not None:
                new_proj.bias = old.bias
        else:
            # otherwise init normally
            nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(old, "bias") and old.bias is not None:
                with torch.no_grad():
                    new_proj.bias.zero_()
    else:
        nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")

    patch_embed.proj = new_proj
    return model


# ------------------------------------------------
# Full Model: SwinUNet + ConvLSTM enhanced decoder
# - Important: encoder takes only the image channel (1). SIM channels are passed
#   separately (we keep compatibility with your current training code which
#   concatenates image + sim and passes them together: forward() will split them).
# ------------------------------------------------
class SwinUNet_ConvLSTM_Enhanced(nn.Module):
    def __init__(self, num_classes, num_seg_classes, sim_depth, use_deconv=True, pretrained=True, img_size=256):
        super().__init__()
        self.N = num_seg_classes
        self.D = sim_depth

        # image channels = 1 (we feed ONLY the image to the Swin encoder)
        in_channels_image = 1

        # --- Encoder (Swin-Base) ---
        self.encoder = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            img_size=img_size
        )
        # adapt patch embedding to accept single-channel image (preserve pretrained by averaging RGB -> single)
        self.encoder = _adapt_swin_input(self.encoder, in_channels_image)

        # SIM channels per timestep (2*N)
        sim_state_ch = 2 * self.N

        # Swin-Base stage channels (channels last from timm, permute to channels-first in forward)
        # For swin_base: stage channels are [128, 256, 512, 1024]
        # We wire decoder to match these sizes:
        self.dec4 = DecoderBlockEnhanced(up_in_ch=1024, skip_ch=512, out_ch=512,
                                        sim_state_channels=sim_state_ch, use_deconv=use_deconv)
        self.dec3 = DecoderBlockEnhanced(up_in_ch=512, skip_ch=256, out_ch=256,
                                        sim_state_channels=sim_state_ch, use_deconv=use_deconv)
        self.dec2 = DecoderBlockEnhanced(up_in_ch=256, skip_ch=128, out_ch=128,
                                        sim_state_channels=sim_state_ch, use_deconv=use_deconv)
        # dec1: no encoder skip at /2; set skip_ch=0 and pass skip=None at call time
        self.dec1 = DecoderBlockEnhanced(up_in_ch=128, skip_ch=0, out_ch=64,
                                        sim_state_channels=sim_state_ch, use_deconv=use_deconv)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _split_sim_states_from_sim(self, sim):
        """
        sim: tensor shaped (B, 2*D*N, H, W)
        returns list of D tensors each (B, 2N, H, W)
        """
        B, C, H, W = sim.shape
        assert C == 2 * self.D * self.N, f"SIM channels mismatch: got {C}, expected {2*self.D*self.N}"
        sim_states = sim.view(B, self.D, 2 * self.N, H, W)
        return [sim_states[:, t] for t in range(self.D)]

    def _resize_states(self, states, size_hw):
        return [F.interpolate(s, size=size_hw, mode="bilinear", align_corners=False) for s in states]

    def forward(self, x):
        """
        Input x: keeps backward compatibility with your training code which concatenates:
          x = torch.cat([image, sim], dim=1)  -> shape (B, 1 + 2*D*N, H, W)
        We split image and sim here:
          image = x[:, :1, :, :]
          sim   = x[:, 1:, :, :]
        """
        # split image and sim
        if x.shape[1] == 1:
            # no SIM provided, create zero sim (useful for quick debugging)
            image = x
            sim = torch.zeros((x.shape[0], 2 * self.D * self.N, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        else:
            image = x[:, :1, :, :]
            sim = x[:, 1:, :, :]

        # Build SIM states list from sim tensor
        sim_states = self._split_sim_states_from_sim(sim)

        # Encoder: pass only the image to Swin
        feats = self.encoder(image)  # list of features [B, H, W, C] (channels-last from timm)
        if not (isinstance(feats, (list, tuple)) and len(feats) >= 4):
            raise RuntimeError("Unexpected encoder feature output. Expected list/tuple of 4 feature tensors.")

        # timm returns [B, H, W, C] -> convert to channels-first (and choose stages)
        e0, e1, e2, e3 = [f.permute(0, 3, 1, 2) for f in feats]  # e0:/4=128, e1:/8=256, e2:/16=512, e3:/32=1024

        # Prepare sim states resized to appropriate spatial sizes
        sim_for_dec4 = self._resize_states(sim_states, e2.shape[-2:])  # e2 spatial size (/16)
        sim_for_dec3 = self._resize_states(sim_states, e1.shape[-2:])  # (/8)
        sim_for_dec2 = self._resize_states(sim_states, e0.shape[-2:])  # (/4)
        # dec1 outputs at /2 (half the input resolution). Resize SIM states to match dec1 spatial size:
        sim_for_dec1 = self._resize_states(sim_states, (image.shape[-2] // 2, image.shape[-1] // 2))

        # Decoder wiring (match channels as set in __init__)
        d4 = self.dec4(e3, e2, sim_for_dec4)     # in 1024 -> out 512 (/16)
        d3 = self.dec3(d4, e1, sim_for_dec3)     # in 512  -> out 256 (/8)
        d2 = self.dec2(d3, e0, sim_for_dec2)     # in 256  -> out 128 (/4)
        d1 = self.dec1(d2, None, sim_for_dec1)   # dec1 has no skip; pass None -> out 64 (/2)

        out = self.final_up(d1)  # -> (B,64,H,W)
        # restore to input resolution
        if out.shape[-2:] != image.shape[-2:]:
            out = F.interpolate(out, size=image.shape[-2:], mode="bilinear", align_corners=False)

        logits = self.head(out)  # (B, num_classes, H, W)
        return logits



