import os
import time
import torch
import torchaudio
import auraloss
from torch import nn
from preprocess import PreProcess
from torch.utils.tensorboard import SummaryWriter
from models import MODEL2, MODEL1

# neural network architecture:
# GLU MLP - biquad filters - GLU MLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


# MODIFIABLE
# ------------------------------------------------
# choose to retrain an existing model or train a new model (True or False)
retrain = False
# select model (MODEL1 or MODEL2)
model_train = "MODEL1"
# select training data (la2a, facebender-rndamp, mcomp-rndamp-A1msR1000ms, ...)
data = "facebender-rndamp"
# select folder name (used in eval.py)
directory = "facebender-rndamp_small_MODEL1"
# GLU MLP hidden layer sizes
layers = [3, 4, 5]
# FC layer size in MODEL2
layer = 5
# number of biquads
n = 5
# length for estimated FIR filter length
N = 32768
# samples used for calculating the loss
seq_length = 1024
# samples used for dividing the audio
# (seq_length and trunc_length should sum to a multiple of N)
# (1*N -> no overlap-add method)
trunc_length = 1 * N - seq_length
batch_size = 50
learning_rate = 1e-3
# loss functions
loss_func = nn.MSELoss()
mr_stft = False
loss_func2 = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 512, 256],
    hop_sizes=[120, 50, 25],
    win_lengths=[600, 240, 100],
    mag_distance="L1",
)
alpha = 0.001
# number of epochs
n_epochs = 3
# ------------------------------------------------


# create folder
if not retrain:
    os.mkdir("results/" + directory)
# create parameters file
with open("results/" + directory + "/parameters.txt", "w") as f:
    f.write(
        f"data: {data}\n"
        + f"model_train: {model_train}\n"
        + "layers: "
        + str(layers).replace(" ", "")
        + f"\nlayer: {layer}\n"
        + f"n: {n}\n"
        + f"N: {N}\n"
        + f"seq_length: {seq_length}\n"
        + f"trunc_length: {trunc_length}\n"
        + f"batch_size: {batch_size}\n"
        + f"learning_rate: {learning_rate}\n"
        + f"MR-STFT: {mr_stft}\n"
        + f"alpha: {alpha}\n"
        + f"epochs: {n_epochs}\n"
        + f"log dir: results/{directory}/model_{data}\n"
        + f"retrain: {retrain}\n"
    )
print("Model: " + directory)

# initialize TensorBoard
print("Initializing TensorBoard")
writer = SummaryWriter(log_dir="results/" + directory + "/" + "model_" + data)

# preprocess audio
train_input, fs = torchaudio.load("data/train/" + data + "-input.wav")
train_target, fs = torchaudio.load("data/train/" + data + "-target.wav")
val_input, fs = torchaudio.load("data/val/" + data + "-input.wav")
val_target, fs = torchaudio.load("data/val/" + data + "-target.wav")
# DataLoader
print("Preprocessing audio (train)")
start = time.time()
train_dataset = PreProcess(
    train_input.float(), train_target.float(), seq_length, trunc_length, batch_size
)
print(f"Time elapsed: {time.time() - start:3.1f}s")
print("Preprocessing audio (val)")
start = time.time()
val_dataset = PreProcess(
    val_input.float(), val_target.float(), seq_length, trunc_length, batch_size
)
print(f"Time elapsed: {time.time() - start:3.1f}s")

# initialize model
print("Initializing model")
if model_train == "MODEL1":
    model = MODEL1(layers, n, N).to(device).train(True)
if model_train == "MODEL2":
    model = MODEL2(layers, layer, n, N).to(device).train(True)
model_optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
)
if retrain:
    model.load_state_dict(torch.load("results/" + directory + "/model.pth"))
    model_optimizer.load_state_dict(
        torch.load("results/" + directory + "/model_optimizer.pth")
    )


# trining and validation loops
def train_loop():
    train_loss = 0
    for X, y in train_dataset:
        X_in = X.to(device)
        y_out = y.to(device)
        # reset gradient
        model_optimizer.zero_grad()

        # compute prediction
        y_hat = model(X_in)

        # calculate loss
        # truncate before to stabilize filters
        if mr_stft:
            loss = loss_func(
                y_hat[:, trunc_length:, 0], y_out[:, trunc_length:, 0]
            ) + alpha * loss_func2(
                y_hat[:, trunc_length:, :].permute(0, 2, 1),
                y_out[:, trunc_length:, :].permute(0, 2, 1),
            )
        else:
            loss = loss_func(y_hat[:, trunc_length:, 0], y_out[:, trunc_length:, 0])

        # backpropagation
        loss.backward()
        model_optimizer.step()

        # accumulate loss
        train_loss += loss.item()

    # return average loss of one epoch
    return train_loss / len(train_dataset)


def val_loop():
    val_loss = 0
    for X, y in val_dataset:
        X_in = X.to(device)
        y_out = y.to(device)
        y_hat = model(X_in)

        if mr_stft:
            loss = loss_func(
                y_hat[:, trunc_length:, 0], y_out[:, trunc_length:, 0]
            ) + alpha * loss_func2(
                y_hat[:, trunc_length:, :].permute(0, 2, 1),
                y_out[:, trunc_length:, :].permute(0, 2, 1),
            )
        else:
            loss = loss_func(y_hat[:, trunc_length:, 0], y_out[:, trunc_length:, 0])

        val_loss += loss.item()

    return val_loss / len(val_dataset)

print("Starting training")

if retrain:
    with torch.no_grad():
        best_loss = val_loop()
else:
    best_loss = float("inf")

start = time.time()
for epoch in range(n_epochs):
    start_epoch = time.time()
    print(f"Epoch {epoch+1}/{n_epochs}")
    # train for one epoch
    model.train(True)
    train_loss = train_loop()

    # validation
    if epoch % 2 == 0:
        with torch.no_grad():
            val_loss = val_loop()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "results/" + directory + "/model.pth")
            torch.save(
                model_optimizer.state_dict(),
                "results/" + directory + "/model_optimizer.pth",
            )
            print("new best val loss")
        print(f"-- Train Loss {train_loss:3E} Val Loss {val_loss:3E}")
    else:
        print(f"-- Train Loss {train_loss:3E}")

    # log loss
    writer.add_scalars(
        "Losses", {"Train loss": train_loss, "Val loss": val_loss}, epoch
    )
    end_epoch = time.time()
    print(f"Epoch time elapsed: {end_epoch - start_epoch:3.1f}s")
writer.flush()
print(f"Total time elapsed: {time.time() - start:3.1f}s")

# save final model for retrain
torch.save(model.state_dict(), "results/" + directory + "/model_final.pth")
torch.save(
    model_optimizer.state_dict(), "results/" + directory + "/model_optimizer_final.pth"
)
