import os
# Check if running on Linux, as ai_edge_torch is only supported on Linux
if os.name != 'posix':
    print("This script is intended to be run on Linux")
    exit()

import torch
import torchaudio
import ai_edge_torch
import time
import numpy as np
from models import MODEL2, MODEL1
from preprocess import PreProcess

# MODIFIABLE
# ------------------------------------------------
# select model to evaluate (directory name)
# directory = "exciter-thick-and-fuzzy_small_MODEL1"
directory = "facebender-rndamp_small_MODEL1"
# ------------------------------------------------

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

params = []
with open(CURRENT_PATH + "/results/" + directory + "/parameters.txt", "r") as file:
    for i, line in enumerate(file.readlines()):
        tmp = line.split()
        if i == 0:
            data = tmp[-1]
        else:
            params.append(tmp[-1])
print("Model: " + directory)

seq_length = int(params[5])
trunc_length = int(params[6])
batch_size = int(params[7])
layers = [int(i) for i in params[1].strip("[]").split(",")]
layer = int(params[2])
n = int(params[3])
if params[0] == "MODEL1":
    model = MODEL1(layers, n, int(params[4]))
else:
    model = MODEL2(layers, layer, n, int(params[4]))
# select proper device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(CURRENT_PATH + "/results/" +
                      directory + "/model.pth", map_location=device))
# save torch model structure if it does not exist
model_path = CURRENT_PATH + "/results/" + directory + "/model.pt"
if not os.path.exists(model_path):
    torch.save(model, model_path)

# get sample input
test_input, fs = torchaudio.load(
    CURRENT_PATH + "/data/test/" + data + "-input.wav")
print("Preprocessing audio")
start = time.time()
test_dataset = PreProcess(
    test_input.float(), test_input.float(), seq_length, trunc_length, batch_size
)
print(f"Time elapsed: {time.time() - start:3.1f}s")
test_x, _ = next(iter(test_dataset))
random_idx = np.random.randint(0, test_x.shape[0])
sample_input = (test_x[random_idx].view(1, -1, 1),)
# sample_input = (test_input.view(1, -1, 1),)

# convert model to Edge
print("converting model to TFlite")
start = time.time()
edge_model = ai_edge_torch.convert(model.eval(), sample_input)
print(f"Time elapsed: {time.time() - start:3.1f}s")

# validate converted model
model.eval()
torch_output = model(*sample_input)
edge_output = edge_model(*sample_input)
if (np.allclose(
    torch_output.detach().numpy(),
    edge_output,
    atol=1e-5,
    rtol=1e-5,
)):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Output of the TFLite conversion doesn't match Pytorch's'")

# save converted model
edge_model.export("results/" + directory + "/tflite_model.tflite")
