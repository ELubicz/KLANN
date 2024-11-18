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
# TODO: this is needed for WSL. Investigate why CUDA is not working
os.environ["PJRT_DEVICE"] = "CPU"

params = []
with open(CURRENT_PATH + "/results/" + directory + "/parameters.txt", "r") as file:
    for i, line in enumerate(file.readlines()):
        tmp = line.split()
        if i == 0:
            data = tmp[-1]
        else:
            params.append(tmp[-1])
print("Model: " + directory)

model_name = params[0]
seq_length = int(params[5])
trunc_length = int(params[6])
batch_size = int(params[7])
layers = [int(i) for i in params[1].strip("[]").split(",")]
layer = int(params[2])
num_biquads = int(params[3])
fir_length = int(params[4])
# Here we define the minimum blocksize. We should be able to process generic blocksizes
# multiple of this one
blocksize = 16
if model_name == "MODEL1":
    model_ref = MODEL1(layers, num_biquads, fir_length)
    model_cmp = MODEL1(layers, num_biquads, fir_length, "custom")
else:
    model_ref = MODEL2(layers, layer, num_biquads, fir_length)
    model_ref = MODEL2(layers, layer, num_biquads, fir_length, "custom")
# select proper device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state_dict = torch.load(CURRENT_PATH + "/results/" +
                              directory + "/model.pth", map_location=device)
model_ref.load_state_dict(model_state_dict)
model_cmp.load_state_dict(model_state_dict)
# save torch model structure if it does not exist
model_path = CURRENT_PATH + "/results/" + directory + "/model.pt"
if not os.path.exists(model_path):
    torch.save(model_ref, model_path)

# get sample input
test_input, fs = torchaudio.load(
    CURRENT_PATH + "/data/test/" + data + "-input.wav")
start = time.time()
# Split audio into multiple blocks of "blocksize"
test_input_blk = np.reshape(test_input, [-1, test_input.shape[0], blocksize])
random_idx = np.random.randint(0, test_input_blk.shape[0])
sample_input_mlp1 = (test_input_blk[random_idx].view(1, -1, 1),)

# convert model to Edge
print("converting model to TFlite")
start = time.time()

edge_model = ai_edge_torch.convert(model_cmp.eval(), sample_input_mlp1)
edge_output = edge_model(*sample_input_mlp1)
edge_model.export("results/" + directory + f"/{model_name}.tflite")

print(f"Time elapsed: {time.time() - start:3.1f}s")

# validate converted model
model_ref.eval()
torch_output = model_ref(*sample_input_mlp1)
print("- Comparing PyTorch x Edge (single block)")
if (np.allclose(torch_output.detach().numpy(), edge_output, atol=2e-5, rtol=2e-5,)):
    print("SUCCESS")
else:
    print("FAILURE")

# TODO: implement block by block appraoch with itermediate saves and compare the filtering of the whole file
