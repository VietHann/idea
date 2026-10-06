"""center=False Hann STFT with explicit causal boundary padding and WOLA."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalSTFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.left_pad = n_fft - hop_length
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)

    def analysis(self, wav):
        if wav.ndim != 3 or wav.size(-1) == 0:
            raise ValueError('Expected nonempty waveform [B,M,N]')
        wav = wav.float()
        right = (-wav.size(-1)) % self.hop_length
        padded = F.pad(wav, (self.left_pad, right))
        frames = padded.unfold(-1, self.n_fft, self.hop_length)
        spectra = torch.fft.rfft(frames * self.window, n=self.n_fft)
        return spectra.permute(0, 2, 3, 1).contiguous()

    def synthesis(self, spectrum, length):
        if spectrum.ndim != 4:
            raise ValueError('Expected spectrum [B,K,T,F]')
        b, k, t, _ = spectrum.shape
        frames = torch.fft.irfft(spectrum.to(torch.complex64), n=self.n_fft) * self.window
        total = (t - 1) * self.hop_length + self.n_fft
        columns = frames.reshape(b * k, t, self.n_fft).transpose(1, 2)
        out = F.fold(columns, (1, total), (1, self.n_fft), stride=(1, self.hop_length)).reshape(b, k, total)
        win2 = self.window.square().view(1, self.n_fft, 1).expand(1, self.n_fft, t)
        norm = F.fold(win2, (1, total), (1, self.n_fft), stride=(1, self.hop_length)).reshape(1, 1, total)
        out = out / norm.clamp_min(1e-8)
        return out[..., self.left_pad:self.left_pad + length]

    def forward(self, wav):
        return self.analysis(wav)

    @property
    def algorithmic_latency_samples(self):
        # Worst-case sample latency for hop-sized packet output. WOLA sample
        # lookahead ranges from n_fft-hop to n_fft, depending on hop position.
        return self.n_fft
