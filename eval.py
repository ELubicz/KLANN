import torch
import torchaudio
import numpy as np
import auraloss
from models import MODEL2, MODEL1

# MODIFIABLE
# ------------------------------------------------
# select model to evaluate (directory name)
directory = "facebender-rndamp_small_MODEL1_test2"
# ------------------------------------------------


params = []
file = open("results/" + directory + "/parameters.txt", "r")
for i, line in enumerate(file.readlines()):
    if i <= 5:
        tmp = line.split()
        if i == 0:
            data = tmp[-1]
        else:
            params.append(tmp[-1])
file.close()
print("Model: " + directory)

layers = [int(i) for i in params[1].strip("[]").split(",")]
layer = int(params[2])
n = int(params[3])
if params[0] == "MODEL1":
    model = MODEL1(layers, n, int(params[4]))
else:
    model = MODEL2(layers, layer, n, int(params[4]))
model.load_state_dict(torch.load("results/" + directory + "/model.pth"))

train_input, fs = torchaudio.load("data/test/" + data + "-input.wav")
train_target, fs = torchaudio.load("data/test/" + data + "-target.wav")


def ESRloss(predict, target):
    return (target - predict).pow(2).sum(dim=-1) / (target.pow(2).sum(dim=-1) + 1e-8)


with torch.no_grad():
    model.eval()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 512, 256],
        hop_sizes=[120, 50, 25],
        win_lengths=[600, 240, 100],
        mag_distance="L1",
    )
    l1loss = torch.nn.L1Loss()
    mseloss = torch.nn.MSELoss()

    y_hat = model((train_input.view(1, -1, 1))).squeeze(-1)

    print(f"L1 loss: {l1loss(y_hat, train_target):.3E}")
    print(f"MSE loss: {mseloss(y_hat, train_target):.3E}")
    ESR_loss = ESRloss(y_hat, train_target)
    print(f"ESR loss: {ESR_loss.item():.3E}")
    print(f"ESR loss dB: {(10 * torch.log10(ESR_loss)).item():.3E}")
    print(
        f"MR-STFT loss: {mrstft(y_hat.reshape(1, 1, -1), train_target.reshape(1, 1, -1)):.3E}"
    )

    torchaudio.save("results/" + directory + "/" + directory + "_hat.wav", y_hat, fs)

# count model parameters
# (layers reversed inside MODEL1/MODEL2 -> reverse back to original)
layers.reverse()
p = (1 + 1) * (2 * layers[0])
for i in range(1, len(layers)):
    p += (1 + layers[i - 1]) * (2 * layers[i])
p += (1 + layers[-1]) * (n)
layers.reverse()
p += (n + 1) * (2 * layers[0])
for i in range(1, len(layers)):
    p += (1 + layers[i - 1]) * (2 * layers[i])
p += (1 + layers[-1]) * (1)
p += 5 * n
if params[0] == "MODEL1":
    print("parameter amount (modedl1): " + str(p))
else:
    p += (n - 1) * ((2 + 1) * (2 * layer) + (1 + layer) * (1))
    print("parameter amount (modedl2): " + str(p))
