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








