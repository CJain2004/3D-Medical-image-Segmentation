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
        self.swin = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        swin_ch = self.swin.feature_info.channels()  # list like [96,192,384,768]
        print("⚙️ Swin encoder channels:", swin_ch)

        # SIM encoder -> match first stage channels
        self.sim_encoder = nn.Conv2d(self.sim_channels, swin_ch[0], kernel_size=1)

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
                print(f"ℹ️ Permuted feat[{i}] from (B,H,W,C) -> (B,C,H,W)")
            else:
                # fallback: try to adapt channels by 1x1 conv or cropping/padding
                if f.shape[1] < expected_c:
                    # pad channels
                    diff = expected_c - f.shape[1]
                    pad = torch.zeros(f.shape[0], diff, f.shape[2], f.shape[3], device=f.device, dtype=f.dtype)
                    fixed.append(torch.cat([f, pad], dim=1))
                    print(f"ℹ️ Padded feat[{i}] channels {f.shape[1]} -> {expected_c}")
                else:
                    # crop
                    fixed.append(f[:, :expected_c, :, :].contiguous())
                    print(f"ℹ️ Cropped feat[{i}] channels {f.shape[1]} -> {expected_c}")
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
        e1 = e1 + sim_feat

        # ---- ConvLSTM at bottleneck ----
        # Ensure e4 has channel count ch4
        ch4 = self.chs[3]
        if e4.shape[1] != ch4:
            # if channels on last dim equal ch4, permute (defensive)
            if e4.shape[-1] == ch4:
                e4 = e4.permute(0, 3, 1, 2).contiguous()
                print("ℹ️ Permuted e4 to (B,C,H,W)")
            else:
                # crop or pad to ch4
                if e4.shape[1] > ch4:
                    e4 = e4[:, :ch4, :, :].contiguous()
                    print(f"ℹ️ Cropped e4 channels to {ch4}")
                else:
                    pad = torch.zeros(e4.shape[0], ch4 - e4.shape[1], e4.shape[2], e4.shape[3], device=e4.device, dtype=e4.dtype)
                    e4 = torch.cat([e4, pad], dim=1)
                    print(f"ℹ️ Padded e4 channels to {ch4}")

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


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    num_classes = 2
    num_seg_classes = 2
    sim_depth = 3
    H, W = 256, 256

    model = SwinUNet_ConvLSTM_SIM(
        num_classes=num_classes,
        num_seg_classes=num_seg_classes,
        sim_depth=sim_depth,
        pretrained=True  
    )
    model.eval()

    img = torch.randn(1, 3, H, W)
    sim = torch.zeros(1, 2 * sim_depth * num_seg_classes, H, W)
    state = None

    with torch.no_grad():
        logits, state = model(img, sim, state)

    print("Input image:", img.shape)
    print("SIM tensor:", sim.shape)
    print("Output logits:", logits.shape)
    if state is not None:
        h, c = state
        print("ConvLSTM hidden:", h.shape)
        print("ConvLSTM cell:", c.shape)
