"""Waveform streaming with persistent analysis, model, and synthesis states."""
import torch
from .utils import detach_tree


class StreamingSession:
    def __init__(self, model, batch_size=1, doa=None, extraction=False):
        if model.training:
            raise ValueError('StreamingSession is inference-only; call model.eval()')
        self.model, self.batch_size, self.doa, self.extraction = model, batch_size, doa, extraction
        c = model.config
        self.device = next(model.parameters()).device
        self.buffer = torch.zeros(batch_size, c.num_channels, c.n_fft - c.hop_length, device=self.device)
        self.numerator = torch.zeros(batch_size, c.num_sources, c.n_fft, device=self.device)
        self.denominator = torch.zeros(c.n_fft, device=self.device)
        self.state = None
        self.input_samples = 0
        self.output_samples = 0
        self.discard = c.n_fft - c.hop_length
        self.closed = False

    def _trim(self, output):
        discard = min(self.discard, output.size(-1))
        self.discard -= discard
        output = output[..., discard:]
        self.output_samples += output.size(-1)
        return output

    def _frame(self, frame):
        c, win = self.model.config, self.model.stft.window
        y = torch.fft.rfft(frame.float() * win, n=c.n_fft).transpose(1, 2)
        result = self.model.step_spectrum(y, self.state, self.doa, extraction=self.extraction)
        self.state = detach_tree(result['state'])
        wave = torch.fft.irfft(result['post_refiner'], n=c.n_fft) * win
        self.numerator += wave
        self.denominator += win.square()
        chunk = self.numerator[..., :c.hop_length] / self.denominator[:c.hop_length].clamp_min(1e-8)
        chunk = chunk.clone()
        self.numerator = torch.nn.functional.pad(self.numerator[..., c.hop_length:], (0, c.hop_length))
        self.denominator = torch.nn.functional.pad(self.denominator[c.hop_length:], (0, c.hop_length))
        return self._trim(chunk)

    @torch.inference_mode()
    def push(self, waveform):
        if self.closed:
            raise RuntimeError('Cannot push after flush')
        if waveform.shape[:2] != (self.batch_size, self.model.config.num_channels):
            raise ValueError('Chunk must be [B,M,N] with session batch/channel count')
        waveform = waveform.to(self.device)
        self.input_samples += waveform.size(-1)
        self.buffer = torch.cat([self.buffer, waveform], dim=-1)
        c, outputs = self.model.config, []
        while self.buffer.size(-1) >= c.n_fft:
            outputs.append(self._frame(self.buffer[..., :c.n_fft]))
            self.buffer = self.buffer[..., c.hop_length:]
        return torch.cat(outputs, dim=-1) if outputs else self.numerator[..., :0]

    @torch.inference_mode()
    def flush(self):
        if self.closed:
            raise RuntimeError('Session already flushed')
        self.closed = True
        c, outputs = self.model.config, []
        right = (-self.input_samples) % c.hop_length
        if right:
            self.buffer = torch.nn.functional.pad(self.buffer, (0, right))
        while self.buffer.size(-1) >= c.n_fft:
            outputs.append(self._frame(self.buffer[..., :c.n_fft]))
            self.buffer = self.buffer[..., c.hop_length:]
        remaining = self.input_samples - self.output_samples
        tail = self.numerator / self.denominator.clamp_min(1e-8)
        discard = min(self.discard, tail.size(-1))
        self.discard -= discard
        tail = tail[..., discard:]
        outputs.append(tail)
        # Padded final hops may include extra samples; trim only the final return.
        output = torch.cat(outputs, dim=-1)
        wanted = self.input_samples - (self.output_samples - sum(x.size(-1) for x in outputs[:-1]))
        output = output[..., :max(0, wanted)]
        self.output_samples = self.input_samples
        return output
