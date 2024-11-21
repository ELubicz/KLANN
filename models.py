import math
import torch
from torch import nn
from torchaudio import functional
from scipy import signal


# state variable filter (SVF) trained in frequency domain and inferred in time domain
class DSVF(nn.Module):
    def __init__(self, N, impl="torchaudio", dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.impl = impl
        # filter parameters
        self.g = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.R = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.m_hp = nn.Parameter(torch.ones(1, dtype=dtype))
        self.m_bp = nn.Parameter(torch.ones(1, dtype=dtype))
        self.m_lp = nn.Parameter(torch.ones(1, dtype=dtype))

        # parameters for the STFT and the overlap-add method
        self.N = N
        self.nfft = 2 ** math.ceil(math.log2(2*self.N-1))

    # x.shape -> (batch_size, time)
    def forward(self, x):
        # restrict frequencies to (0, pi)
        g = torch.tan(torch.pi*1/(1+torch.exp(-self.g))/2)
        # restrict R to > 0
        R = nn.functional.softplus(self.R)
        g_2 = g * g
        b = torch.cat((g_2 * self.m_lp + g * self.m_bp + self.m_hp,
                       2 * g_2 * self.m_lp - 2 * self.m_hp,
                       g_2 * self.m_lp - g * self.m_bp + self.m_hp), dim=0)
        a = torch.cat((g_2 + 2 * R * g + 1,
                       2 * g_2 - 2,
                       g_2 - 2 * R * g + 1), dim=0)

        # filter in frequency domain
        if self.training:
            # divide x into sub-frames of length N and perform convolution in frequency domain
            # (length of x must be divisible by N)
            segments = x.view(x.shape[0], -1, self.N)
            X = torch.fft.rfft(segments, n=self.nfft, dim=-1)
            H = torch.fft.rfft(b, n=self.nfft, dim=-1) / \
                torch.fft.rfft(a, n=self.nfft, dim=-1)
            y = torch.fft.irfft(X * H, n=self.nfft, dim=-1)

            if segments.shape[1] == 1:
                return y[:, :, 0:self.N].flatten(-2)
            else:
                # overlap-add
                firstPart = y[:, :, 0:self.N]
                overlap = y[:, :-1, self.N:2*self.N]
                overlapExt = nn.functional.pad(
                    overlap, (0, 0, 1, 0), "constant", 0)  # pad the first frame
                return (firstPart + overlapExt).flatten(-2)

        # filter in time domain
        else:
            with torch.no_grad():
                orig_type = x.dtype
                x = x.type(self.dtype)
                if self.impl == "torchaudio":
                    output = functional.lfilter(
                        x.type(self.dtype), a, b, clamp=False)
                elif self.impl == "scipy":
                    # Note that due to precision errors, this will only be equivalent to the torchaudio implementation
                    # if the filter coefficients are of type float64, as well as the input data.
                    # tourchaudio implements DFI and scipy DFII-transposed (https://github.com/pytorch/audio/issues/2500).
                    # This accounts for the precision differences when using float32.
                    output = torch.tensor(signal.lfilter(b, a, x))
                elif self.impl == "custom":
                    '''
                    Here we have the pure Python implementation of the lfilter function
                    as per the implementation in torchaudio.functional.filtering.
                    see _lfilter_core_generic_loop() and _lfilter_core
                    '''
                    a_coeffs = a.unsqueeze(0)
                    b_coeffs = b.unsqueeze(0)
                    shape = x.size()
                    x = x.reshape(-1, a_coeffs.shape[0], shape[-1])
                    # output = _lfilter(waveform, a_coeffs, b_coeffs)
                    n_batch, n_channel, n_sample = x.size()
                    n_order = a_coeffs.size(1)
                    padded_waveform = torch.nn.functional.pad(
                        x, [n_order - 1, 0])
                    padded_output_waveform = torch.zeros_like(padded_waveform)

                    # Set up the coefficients matrix
                    # Flip coefficients' order
                    a_coeffs_flipped = a_coeffs.flip(1)
                    b_coeffs_flipped = b_coeffs.flip(1)

                    # calculate windowed_input_signal in parallel using convolution
                    input_signal_windows = torch.nn.functional.conv1d(
                        padded_waveform, b_coeffs_flipped.unsqueeze(1), groups=n_channel)

                    input_signal_windows.div_(a_coeffs[:, :1])
                    a_coeffs_flipped.div_(a_coeffs[:, :1])

                    # _lfilter_core_generic_loop(input_signal_windows, a_coeffs_flipped, padded_output_waveform)
                    n_order = a_coeffs_flipped.size(1)
                    a_coeffs_flipped = a_coeffs_flipped.unsqueeze(2)
                    for i_sample, o0 in enumerate(input_signal_windows.permute(2, 0, 1)):
                        windowed_output_signal = padded_output_waveform[:,
                                                                        :, i_sample: i_sample + n_order]
                        o0 -= (windowed_output_signal.transpose(0, 1)
                               @ a_coeffs_flipped)[..., 0].t()
                        padded_output_waveform[:, :,
                                               i_sample + n_order - 1] = o0

                    output = padded_output_waveform[:, :, n_order - 1:]
                    # unpack batch
                    output = output.reshape(shape[:-1] + output.shape[-1:])

                return output.type(orig_type)


class MODEL_BASE(nn.Module):
    def __init__(self, hidden_layer_sizes, num_biquads, 
                 fft_length, impl="torchaudio", dtype=torch.float32):
        super().__init__()
        self.n = num_biquads
        mlp1 = []
        mlp1.append(nn.Linear(1, 2*hidden_layer_sizes[0]))
        mlp1.append(nn.GLU())
        for i in range(1, len(hidden_layer_sizes)):
            mlp1.append(
                nn.Linear(hidden_layer_sizes[i-1], 2*hidden_layer_sizes[i]))
            mlp1.append(nn.GLU())
        mlp1.append(nn.Linear(hidden_layer_sizes[-1], num_biquads))
        self.mlp1 = nn.Sequential(*mlp1)

        self.filters = nn.ModuleList([])
        for _ in range(self.n):
            self.filters.append(DSVF(fft_length, impl, dtype))

        reverse_layers = list(hidden_layer_sizes)
        reverse_layers.reverse()
        mlp2 = []
        mlp2.append(nn.Linear(num_biquads, 2*reverse_layers[0]))
        mlp2.append(nn.GLU())
        for i in range(1, len(reverse_layers)):
            mlp2.append(nn.Linear(reverse_layers[i-1], 2*reverse_layers[i]))
            mlp2.append(nn.GLU())
        mlp2.append(nn.Linear(reverse_layers[-1], 1))
        self.mlp2 = nn.Sequential(*mlp2)


# DSVFs in parallel
class MODEL1(MODEL_BASE):
    def __init__(self, layers, num_biquads, fft_length, impl="torchaudio", dtype=torch.float32):
        super().__init__(layers, num_biquads, fft_length, impl, dtype)

    def forward(self, x):
        z = self.mlp1(x)
        y = []
        for i in range(self.n):
            y.append(self.filters[i](z[:, :, i]).unsqueeze(-1))
        return self.mlp2(torch.cat(y, dim=-1))

# DSVFs in parallel and series


class MODEL2(MODEL_BASE):
    def __init__(self, hidden_layer_sizes, fc_layer_size, num_biquads,
                 fft_length, impl="torchaudio", dtype=torch.float32):
        super().__init__(hidden_layer_sizes, num_biquads, fft_length, impl, dtype)

        self.linear = nn.ModuleList([])
        for _ in range(self.n-1):
            self.linear.append(nn.Sequential(
                nn.Linear(2, 2*fc_layer_size),
                nn.GLU(),
                nn.Linear(fc_layer_size, 1)
            ))

    # x -> (batch_size, samples, input_size)
    def forward(self, x):
        y = self.mlp1(x)
        z = self.filters[0](y[:, :, 0]).unsqueeze(-1)
        z_s = []
        z_s.append(z)
        for i in range(self.n-1):
            z = self.filters[i+1](self.linear[i](
                torch.cat((z, y[:, :, i+1].unsqueeze(-1)), dim=-1)).squeeze(-1)).unsqueeze(-1)
            z_s.append(z)
        return self.mlp2(torch.cat(z_s, dim=-1))
    # return -> (batch_size, samples, input_size)
