import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

# -----------------------------
# ConvLSTM (single-layer cell)
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad, bias=bias)

    def forward(self, x, state):
        """
        x:      (B, C_in, H, W)
        state:  (h, c) where each is (B, hidden_dim, H, W)
        """
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        (i, f, o, g) = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, x):
        B, _, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        return h, c


# --------------------------------------------
# Decoder block: Up + Skip + ConvLSTM (guided)
# --------------------------------------------
class DecoderBlockConvLSTM(nn.Module):
    def __init__(self, up_in_ch, skip_ch, out_ch, sim_state_channels, use_deconv=True):
        """
        up_in_ch:    channels from previous (deeper) level
        skip_ch:     channels from encoder skip connection
        out_ch:      output channels of this decoder stage (also ConvLSTM hidden_dim)
        sim_state_channels: channels per SIM state at input (2N)
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2) if use_deconv \
                  else nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     nn.Conv2d(up_in_ch, out_ch, kernel_size=1))
        # After upsample, we will concat with skip, so input to ConvLSTM is out_ch + skip_ch
        self.in_ch = out_ch + skip_ch
        self.hidden_ch = out_ch

        # Project SIM state (2N) to the spatial feature size for this stage
        self.sim_proj = nn.Conv2d(sim_state_channels, self.in_ch, kernel_size=1)

        # ConvLSTM to refine with temporal guidance
        self.clstm = ConvLSTMCell(input_dim=self.in_ch, hidden_dim=self.hidden_ch, kernel_size=3)

        # Optional light conv after ConvLSTM for smoothing
        self.post = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip, sim_states_list):
        """
        x:                (B, up_in_ch, H/2, W/2)  -> upsampled to match skip spatial size
        skip:             (B, skip_ch, H, W)
        sim_states_list:  list of D tensors, each (B, 2N, H, W) for this stage's resolution
        """
        x = self.up(x)
        # Pad if needed due to odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)  # (B, in_ch, H, W)

        # Initialize ConvLSTM state
        h, c = self.clstm.init_state(x)

        # Run through D time steps, injecting per-state SIM guidance
        for sim_state in sim_states_list:
            guided = x + self.sim_proj(sim_state)   # residual guidance
            h, c = self.clstm(guided, (h, c))

        out = self.post(h)
        return out


# ---------------------------------------------------
# ResNet34 Encoder adapted for arbitrary in_channels
# ---------------------------------------------------
def _adapt_resnet34_first_conv(model, in_channels):
    # Replace conv1 to accept arbitrary in_channels (no pretrained weights here)
    old = model.conv1
    model.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


# ---------------------------------------------------
# Full model: ResNet34-UNet with ConvLSTM decoder
# ---------------------------------------------------
class ResNet34_UNet_ConvLSTM(nn.Module):
    def __init__(self, num_classes, num_seg_classes, sim_depth, use_deconv=True):
        """
        Args:
            num_classes      (int): Output classes (== N). Produces logits with N channels.
            num_seg_classes  (int): N (incl. background) — used to parse SIM.
            sim_depth        (int): D — number of SIM states
            use_deconv       (bool): Use transposed conv for upsampling (else bilinear+1x1)
        """
        super().__init__()
        self.N = num_seg_classes
        self.D = sim_depth
        in_channels = 1 + 2 * self.D * self.N  # image + SIM channels

        # --- Encoder (ResNet-34) ---
        enc = resnet34(weights=None)
        enc = _adapt_resnet34_first_conv(enc, in_channels) # adapt first conv to input channels, Normally ResNet34 expects RGB (3 channels).
# Here we replace the first conv with one that takes 1 + 2*D*N channels.

        self.enc_conv1 = nn.Sequential(
            enc.conv1, enc.bn1, enc.relu
        )  # -> C=64, /2
        self.enc_pool = enc.maxpool           # /4
        self.enc_layer1 = enc.layer1          # 64,  /4
        self.enc_layer2 = enc.layer2          # 128, /8
        self.enc_layer3 = enc.layer3          # 256, /16
        self.enc_layer4 = enc.layer4          # 512, /32

        # --- Decoder with ConvLSTM guidance ---
        # SIM state channels per timestep before projection = 2N
        sim_state_ch = 2 * self.N

        self.dec4 = DecoderBlockConvLSTM(up_in_ch=512, skip_ch=256, out_ch=256,
                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)  # /16
        self.dec3 = DecoderBlockConvLSTM(up_in_ch=256, skip_ch=128, out_ch=128,
                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)  # /8
        self.dec2 = DecoderBlockConvLSTM(up_in_ch=128, skip_ch=64,  out_ch=64,
                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)  # /4
        self.dec1 = DecoderBlockConvLSTM(up_in_ch=64,  skip_ch=64,  out_ch=64,
                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)  # /2

        # Final up to original /1 resolution
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    # ---------- SIM utilities ----------
    def _split_sim_states(self, x):
        """
        x: (B, 1 + 2DN, H, W)
        returns list of D tensors, each (B, 2N, H, W)
        """
        B, C, H, W = x.shape
        sim = x[:, 1:, :, :]                     # drop the first image channel
        assert sim.shape[1] == 2 * self.D * self.N, "Input channels don't match 1 + 2*D*N"
        sim_states = sim.view(B, self.D, 2 * self.N, H, W)  # (B, D, 2N, H, W)
        # Newest state is assumed to be at the front (as per our update_sim); keep that order
        states = [sim_states[:, t] for t in range(self.D)]  # list of D x (B, 2N, H, W)
        return states

    def _resize_states(self, states, size_hw):
        """Resize list of (B, 2N, H, W) to target spatial size."""
        resized = [F.interpolate(s, size=size_hw, mode="bilinear", align_corners=False) for s in states]
        return resized

    # ---------- Forward ----------
    def forward(self, x):
        """
        x: (B, 1 + 2DN, H, W)  -> logits: (B, N, H, W)
        """
        # Prepare SIM state list at input resolution (H, W)
        sim_states = self._split_sim_states(x)

        # Encoder
        e0 = self.enc_conv1(x)          # 64,  H/2
        p0 = self.enc_pool(e0)          #      H/4
        e1 = self.enc_layer1(p0)        # 64,  H/4
        e2 = self.enc_layer2(e1)        # 128, H/8
        e3 = self.enc_layer3(e2)        # 256, H/16
        e4 = self.enc_layer4(e3)        # 512, H/32   (bottleneck)

        # Resize SIM states per decoder stage resolution
        sim_16 = self._resize_states(sim_states, e3.shape[-2:])  # for dec4
        sim_8  = self._resize_states(sim_states, e2.shape[-2:])  # for dec3
        sim_4  = self._resize_states(sim_states, e1.shape[-2:])  # for dec2
        sim_2  = self._resize_states(sim_states, e0.shape[-2:])  # for dec1

        # Decoder with ConvLSTM guidance
        d4 = self.dec4(e4, e3, sim_16)  # -> 256, /16
        d3 = self.dec3(d4, e2, sim_8)   # -> 128, /8
        d2 = self.dec2(d3, e1, sim_4)   # -> 64,  /4
        d1 = self.dec1(d2, e0, sim_2)   # -> 64,  /2

        out = self.final_up(d1)         # -> 64,  /1 (match input H,W)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        logits = self.head(out)         # (B, N, H, W)
        return logits
















# # models.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# # -----------------------------
# # ConvLSTM (single-layer)
# # -----------------------------
# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
#         super().__init__()
#         pad = kernel_size // 2
#         self.hidden_dim = hidden_dim
#         self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
#                               kernel_size, padding=pad, bias=bias)

#     def forward(self, x, state):
#         h, c = state
#         combined = torch.cat([x, h], dim=1)
#         gates = self.conv(combined)
#         i, f, o, g = torch.chunk(gates, 4, dim=1)
#         i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
#         g = torch.tanh(g)
#         c_next = f * c + i * g
#         h_next = o * torch.tanh(c_next)
#         return h_next, c_next

#     def init_state(self, x):
#         # x: (B, C, H, W)
#         B, _, H, W = x.shape
#         h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
#         c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
#         return h, c


# # -----------------------------
# # Deform-like Attention Module
# # (lightweight: MultiheadAttention over flattened spatial tokens)
# # -----------------------------
# class DeformableFusion(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         # registered projection layer for skip -> dim (created lazily if needed)
#         self.skip_proj = None
#         self.skip_proj_in_ch = None

#     def forward(self, x, skip):
#         B, C, H, W = x.shape
#         if skip is None:
#             return x

#         # ensure spatial size
#         if skip.shape[-2:] != (H, W):
#             skip = F.interpolate(skip, size=(H, W), mode='bilinear', align_corners=False)

#         # create/register a projection if skip channels != C
#         if skip.shape[1] != C:
#             if (self.skip_proj is None) or (self.skip_proj_in_ch != skip.shape[1]):
#                 # create a new projection and register it
#                 self.skip_proj = nn.Conv2d(skip.shape[1], C, kernel_size=1).to(x.device)
#                 self.skip_proj_in_ch = skip.shape[1]
#             skip = self.skip_proj(skip)

#         q = x.flatten(2).transpose(1, 2)   # (B, HW, C)
#         k = skip.flatten(2).transpose(1, 2)
#         v = k
#         attn_out, _ = self.attn(q, k, v)   # (B, HW, C)
#         attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
#         return self.proj(attn_out + x)


# # -----------------------------
# # Mix FFN (depthwise conv + 1x1 project)
# # -----------------------------
# class MixFFN(nn.Module):
#     def __init__(self, dim, expansion=4, drop=0.0):
#         super().__init__()
#         hidden = dim * expansion
#         self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
#         # depthwise conv for local mixing
#         self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
#         self.act = nn.GELU()
#         self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
#         self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.dwconv(x)
#         x = self.act(x)
#         x = self.drop(self.fc2(x))
#         return x


# # -----------------------------------------
# # Decoder block: Up + Fusion + ConvLSTM + MixFFN
# # -----------------------------------------
# class DecoderBlockEnhanced(nn.Module):
#     def __init__(self, up_in_ch, skip_ch, out_ch, sim_state_channels, use_deconv=True):
#         """
#         up_in_ch: channels of incoming (deeper) feature
#         skip_ch: channels of skip (encoder) feature (0 or None if absent)
#         out_ch: output channels of this block
#         sim_state_channels: channels in each SIM state (2*N)
#         """
#         super().__init__()
#         self.use_deconv = use_deconv
#         if use_deconv:
#             self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
#         else:
#             self.up = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#                 nn.Conv2d(up_in_ch, out_ch, kernel_size=1)
#             )

#         # Fusion module expects channels == out_ch
#         self.fusion = DeformableFusion(dim=out_ch, num_heads=max(1, out_ch // 64))

#         # project sim -> out_ch if sim channels exist
#         self.sim_proj = nn.Conv2d(sim_state_channels, out_ch, kernel_size=1)

#         # ConvLSTM expects input channels == out_ch
#         self.clstm = ConvLSTMCell(input_dim=out_ch, hidden_dim=out_ch, kernel_size=3)

#         self.mix_ffn = MixFFN(out_ch, expansion=4, drop=0.0)

#     def forward(self, x, skip, sim_states_list):
#         """
#         x: (B, up_in_ch, H/2, W/2)
#         skip: (B, skip_ch, H, W) or None
#         sim_states_list: list of D tensors resized to this block's spatial size, each (B, sim_ch, H, W)
#         """
#         x = self.up(x)  # now (B, out_ch, H, W) if ConvTranspose out channels == out_ch
#         # ensure spatial match
#         if skip is not None and x.shape[-2:] != skip.shape[-2:]:
#             x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

#         # fusion: if skip is None, fusion returns x
#         x = self.fusion(x, skip)

#         # ConvLSTM initial state (zeros)
#         h, c = self.clstm.init_state(x)

#         # run through D timesteps injecting SIM guidance
#         for sim in sim_states_list:
#             guided = x + self.sim_proj(sim)
#             h, c = self.clstm(guided, (h, c))

#         out = self.mix_ffn(h)
#         return out


# # ------------------------------------------------
# # Adapt Swin patch embed to accept custom in_channels
# # ------------------------------------------------
# def _adapt_swin_input(model, in_ch):
#     """
#     Replaces patch_embed.proj to accept in_ch input channels.
#     If the original proj had 3-channel pretrained weights and in_ch == 1,
#     initialize new proj by averaging RGB weights -> preserves pretrained signal.
#     """
#     patch_embed = model.patch_embed
#     old = patch_embed.proj

#     # obtain layer attributes (safe fallback)
#     kernel_size = old.kernel_size if hasattr(old, "kernel_size") else (4, 4)
#     stride = old.stride if hasattr(old, "stride") else (4, 4)
#     padding = old.padding if hasattr(old, "padding") else 0
#     out_ch = old.out_channels if hasattr(old, "out_channels") else old.weight.shape[0]

#     new_proj = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

#     # If old has pretrained weights with 3 input channels and we adapt to 1, copy mean(R,G,B)
#     if hasattr(old, "weight"):
#         old_w = old.weight.data
#         if old_w.ndim == 4 and old_w.shape[1] == 3 and in_ch == 1:
#             # mean across input channel dim to produce single-channel kernel
#             new_proj.weight.data = old_w.mean(dim=1, keepdim=True).clone()
#             if hasattr(old, "bias") and old.bias is not None:
#                 new_proj.bias = old.bias
#         else:
#             # otherwise init normally
#             nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")
#             if hasattr(old, "bias") and old.bias is not None:
#                 with torch.no_grad():
#                     new_proj.bias.zero_()
#     else:
#         nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")

#     patch_embed.proj = new_proj
#     return model


# # ------------------------------------------------
# # Full Model: SwinUNet + ConvLSTM enhanced decoder
# # - Important: encoder takes only the image channel (1). SIM channels are passed
# #   separately (we keep compatibility with your current training code which
# #   concatenates image + sim and passes them together: forward() will split them).
# # ------------------------------------------------
# class SwinUNet_ConvLSTM_Enhanced(nn.Module):
#     def __init__(self, num_classes, num_seg_classes, sim_depth, use_deconv=True, pretrained=True, img_size=256):
#         super().__init__()
#         self.N = num_seg_classes
#         self.D = sim_depth

#         # image channels = 1 (we feed ONLY the image to the Swin encoder)
#         in_channels_image = 1

#         # --- Encoder (Swin-Base) ---
#         self.encoder = timm.create_model(
#             'swin_base_patch4_window7_224',
#             pretrained=pretrained,
#             features_only=True,
#             img_size=img_size
#         )
#         # adapt patch embedding to accept single-channel image (preserve pretrained by averaging RGB -> single)
#         self.encoder = _adapt_swin_input(self.encoder, in_channels_image)

#         # SIM channels per timestep (2*N)
#         sim_state_ch = 2 * self.N

#         # Swin-Base stage channels (channels last from timm, permute to channels-first in forward)
#         # For swin_base: stage channels are [128, 256, 512, 1024]
#         # We wire decoder to match these sizes:
#         self.dec4 = DecoderBlockEnhanced(up_in_ch=1024, skip_ch=512, out_ch=512,
#                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)
#         self.dec3 = DecoderBlockEnhanced(up_in_ch=512, skip_ch=256, out_ch=256,
#                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)
#         self.dec2 = DecoderBlockEnhanced(up_in_ch=256, skip_ch=128, out_ch=128,
#                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)
#         # dec1: no encoder skip at /2; set skip_ch=0 and pass skip=None at call time
#         self.dec1 = DecoderBlockEnhanced(up_in_ch=128, skip_ch=0, out_ch=64,
#                                         sim_state_channels=sim_state_ch, use_deconv=use_deconv)

#         self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#         self.head = nn.Conv2d(64, num_classes, kernel_size=1)

#     def _split_sim_states_from_sim(self, sim):
#         """
#         sim: tensor shaped (B, 2*D*N, H, W)
#         returns list of D tensors each (B, 2N, H, W)
#         """
#         B, C, H, W = sim.shape
#         assert C == 2 * self.D * self.N, f"SIM channels mismatch: got {C}, expected {2*self.D*self.N}"
#         sim_states = sim.view(B, self.D, 2 * self.N, H, W)
#         return [sim_states[:, t] for t in range(self.D)]

#     def _resize_states(self, states, size_hw):
#         return [F.interpolate(s, size=size_hw, mode="bilinear", align_corners=False) for s in states]

#     def forward(self, x):
#         """
#         Input x: keeps backward compatibility with your training code which concatenates:
#           x = torch.cat([image, sim], dim=1)  -> shape (B, 1 + 2*D*N, H, W)
#         We split image and sim here:
#           image = x[:, :1, :, :]
#           sim   = x[:, 1:, :, :]
#         """
#         # split image and sim
#         if x.shape[1] == 1:
#             # no SIM provided, create zero sim (useful for quick debugging)
#             image = x
#             sim = torch.zeros((x.shape[0], 2 * self.D * self.N, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
#         else:
#             image = x[:, :1, :, :]
#             sim = x[:, 1:, :, :]

#         # Build SIM states list from sim tensor
#         sim_states = self._split_sim_states_from_sim(sim)

#         # Encoder: pass only the image to Swin
#         feats = self.encoder(image)  # list of features [B, H, W, C] (channels-last from timm)
#         if not (isinstance(feats, (list, tuple)) and len(feats) >= 4):
#             raise RuntimeError("Unexpected encoder feature output. Expected list/tuple of 4 feature tensors.")

#         # timm returns [B, H, W, C] -> convert to channels-first (and choose stages)
#         e0, e1, e2, e3 = [f.permute(0, 3, 1, 2) for f in feats]  # e0:/4=128, e1:/8=256, e2:/16=512, e3:/32=1024

#         # Prepare sim states resized to appropriate spatial sizes
#         sim_for_dec4 = self._resize_states(sim_states, e2.shape[-2:])  # e2 spatial size (/16)
#         sim_for_dec3 = self._resize_states(sim_states, e1.shape[-2:])  # (/8)
#         sim_for_dec2 = self._resize_states(sim_states, e0.shape[-2:])  # (/4)
#         # dec1 outputs at /2 (half the input resolution). Resize SIM states to match dec1 spatial size:
#         sim_for_dec1 = self._resize_states(sim_states, (image.shape[-2] // 2, image.shape[-1] // 2))

#         # Decoder wiring (match channels as set in __init__)
#         d4 = self.dec4(e3, e2, sim_for_dec4)     # in 1024 -> out 512 (/16)
#         d3 = self.dec3(d4, e1, sim_for_dec3)     # in 512  -> out 256 (/8)
#         d2 = self.dec2(d3, e0, sim_for_dec2)     # in 256  -> out 128 (/4)
#         d1 = self.dec1(d2, None, sim_for_dec1)   # dec1 has no skip; pass None -> out 64 (/2)

#         out = self.final_up(d1)  # -> (B,64,H,W)
#         # restore to input resolution
#         if out.shape[-2:] != image.shape[-2:]:
#             out = F.interpolate(out, size=image.shape[-2:], mode="bilinear", align_corners=False)

#         logits = self.head(out)  # (B, num_classes, H, W)
#         return logits



# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -----------------------------
# ConvLSTM Cell (single layer)
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad, bias=bias)

    def init_state(self, x):
        # x: (B, C, H, W)
        return (torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype),
                torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype))

    def forward(self, x, hidden=None):
        """
        x: (B, input_dim, H, W)
        hidden: (h, c) each (B, hidden_dim, H, W) or None
        """
        if hidden is None:
            h, c = self.init_state(x)
        else:
            h, c = hidden
            # spatial align
            if h.shape[2:] != x.shape[2:]:
                h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
                c = F.interpolate(c, size=x.shape[2:], mode='bilinear', align_corners=False)
            # channel align (rare)
            if h.shape[1] != self.hidden_dim:
                proj_h = nn.Conv2d(h.shape[1], self.hidden_dim, kernel_size=1).to(h.device)
                proj_c = nn.Conv2d(c.shape[1], self.hidden_dim, kernel_size=1).to(c.device)
                h = proj_h(h)
                c = proj_c(c)

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# -----------------------------
# SwinUNet + ConvLSTM + SIM
# -----------------------------
class SwinUNet_ConvLSTM_SIM(nn.Module):
    def __init__(self, num_classes, num_seg_classes, sim_depth, pretrained=True,
                 backbone="swinv2_tiny_window8_256"):
        """
        num_classes    : segmentation output channels
        num_seg_classes: N (including background) used by SIM
        sim_depth      : D (number of SIM rounds stored)
        backbone       : timm backbone name (ensure img_size matches your input)
        """
        super().__init__()
        self.N = num_seg_classes
        self.D = sim_depth
        self.sim_channels = 2 * self.D * self.N

        # -------- Swin encoder --------
        self.swin = timm.create_model(backbone, pretrained=pretrained, features_only=True,in_chans=1)
        swin_ch = self.swin.feature_info.channels()  # list like [96,192,384,768]
        print("⚙️ Swin encoder channels:", swin_ch)

        # SIM encoder -> match first stage channels
        self.sim_encoder = nn.Conv2d(self.sim_channels, swin_ch[0], kernel_size=1)
        self.att_gate = nn.Sequential(
            nn.Conv2d(self.swin.feature_info.channels()[0] * 2,  # concatenate e1 + sim_feat
                    self.swin.feature_info.channels()[0],
                    kernel_size=1),
            nn.BatchNorm2d(self.swin.feature_info.channels()[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.swin.feature_info.channels()[0], 1, kernel_size=1)
        )

        # Decoder up layers (simple)
        self.up4 = nn.ConvTranspose2d(swin_ch[3], swin_ch[2], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(swin_ch[2], swin_ch[1], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(swin_ch[1], swin_ch[0], kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(swin_ch[0], num_classes, kernel_size=1)

        # ConvLSTM at bottleneck (use ch4 as both input and hidden dims for addition-based input)
        ch4 = swin_ch[3]
        self.chs = swin_ch  # store for checks
        self.convlstm = ConvLSTMCell(input_dim=ch4, hidden_dim=ch4)

    def _ensure_channel_first(self, feats):
        """
        timm should return (B,C,H,W) for features, but some backbones or versions might return (B,H,W,C).
        This function detects such mismatch and permutes to (B,C,H,W) if needed.
        """
        fixed = []
        for i, f in enumerate(feats):
            if f.dim() != 4:
                raise ValueError(f"Feature {i} has unexpected dim {f.dim()}")
            expected_c = self.chs[i]
            # if channel dimension already matches expected, keep
            if f.shape[1] == expected_c:
                fixed.append(f)
            # else if last dim matches expected, assume (B,H,W,C) and permute
            elif f.shape[-1] == expected_c:
                f2 = f.permute(0, 3, 1, 2).contiguous()
                fixed.append(f2)
                # print(f"ℹ️ Permuted feat[{i}] from (B,H,W,C) -> (B,C,H,W)")
            else:
                # fallback: try to adapt channels by 1x1 conv or cropping/padding
                if f.shape[1] < expected_c:
                    # pad channels
                    diff = expected_c - f.shape[1]
                    pad = torch.zeros(f.shape[0], diff, f.shape[2], f.shape[3], device=f.device, dtype=f.dtype)
                    fixed.append(torch.cat([f, pad], dim=1))
                    # print(f"ℹ️ Padded feat[{i}] channels {f.shape[1]} -> {expected_c}")
                else:
                    # crop
                    fixed.append(f[:, :expected_c, :, :].contiguous())
                    # print(f"ℹ️ Cropped feat[{i}] channels {f.shape[1]} -> {expected_c}")
        return fixed

    def forward(self, x, sim, state=None):
        """
        x:   (B, C, H, W)
        sim: (B, 2*D*N, H, W)
        state: optional (h, c)
        """
        feats = self.swin(x)  # list of feature maps
        feats = self._ensure_channel_first(feats)
        e1, e2, e3, e4 = feats

        # Debug prints to help trace shapes (comment out later if noisy)
        # print("DEBUG e1,e2,e3,e4 shapes:", e1.shape, e2.shape, e3.shape, e4.shape)

        # ---- fuse SIM early into e1 ----
        sim_feat = self.sim_encoder(sim)
        if sim_feat.shape[2:] != e1.shape[2:]:
            sim_feat = F.interpolate(sim_feat, size=e1.shape[2:], mode='bilinear', align_corners=False)
        # align channels (safe)
        if sim_feat.shape[1] != e1.shape[1]:
            if sim_feat.shape[1] > e1.shape[1]:
                sim_feat = sim_feat[:, : e1.shape[1], :, :].contiguous()
            else:
                pad = torch.zeros(sim_feat.shape[0], e1.shape[1] - sim_feat.shape[1], sim_feat.shape[2], sim_feat.shape[3], device=sim_feat.device, dtype=sim_feat.dtype)
                sim_feat = torch.cat([sim_feat, pad], dim=1)

        # attention-based SIM fusion
        att_input = torch.cat([e1, sim_feat], dim=1)
        gate = torch.sigmoid(self.att_gate(att_input))   # (B,1,H,W)
        e1 = e1 + gate * sim_feat

        # ---- ConvLSTM at bottleneck ----
        # Ensure e4 has channel count ch4
        ch4 = self.chs[3]
        if e4.shape[1] != ch4:
            # if channels on last dim equal ch4, permute (defensive)
            if e4.shape[-1] == ch4:
                e4 = e4.permute(0, 3, 1, 2).contiguous()
                # print("ℹ️ Permuted e4 to (B,C,H,W)")
            else:
                # crop or pad to ch4
                if e4.shape[1] > ch4:
                    e4 = e4[:, :ch4, :, :].contiguous()
                    # print(f"ℹ️ Cropped e4 channels to {ch4}")
                else:
                    pad = torch.zeros(e4.shape[0], ch4 - e4.shape[1], e4.shape[2], e4.shape[3], device=e4.device, dtype=e4.dtype)
                    e4 = torch.cat([e4, pad], dim=1)
                    # print(f"ℹ️ Padded e4 channels to {ch4}")

        # Debug shapes before ConvLSTM
        # print(f"Before ConvLSTM: e4 {tuple(e4.shape)}; state given {state is not None}")

        h, c = self.convlstm(e4, state)  # ConvLSTMCell handles None state or mismatched spatial sizes

        # ---- decode: simple upsample + skip-add fusion ----
        d4 = self.up4(h)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = d4 + e3

        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + e2

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + e1

        out = self.out_conv(d2)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out, (h, c)

