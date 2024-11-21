import os
import torch
import torchaudio
import time
import numpy as np
import models
import models_tf
from preprocess import PreProcess
import tensorflow as tf

# MODIFIABLE
# ------------------------------------------------
# select model to evaluate (directory name)
# directory = "exciter-thick-and-fuzzy_small_MODEL1"
directory = "facebender-rndamp_small_MODEL1"
# ------------------------------------------------


def copy_weights(model_torch, model_tf):
    """
    Copy weights from a torch model to a tf model
    Assumes that the models have the same architecture, layered in the same order
    """
    # First some dummy checks
    torch_state_dict = model_torch.state_dict()
    torch_weights = list(torch_state_dict.values())
    tf_weights = model_tf.get_weights()
    assert len(torch_weights) == len(tf_weights)
    for tf_weight, torch_weight in zip(tf_weights, torch_weights):
        assert np.prod(tf_weight.shape) == np.prod(torch_weight.shape)
    # Now we can copy the weights
    for tf_layer in model_tf.layers:
        for tf_weight in tf_layer.weights:
            # pop the first torch wight from the list
            torch_weight = torch_weights.pop(0)
            if "kernel" in tf_weight.name:
                tf_weight.assign(torch_weight.detach().numpy().T)
            else:
                tf_weight.assign(torch_weight.detach().numpy())


def main():
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

    fft_length = int(params[4])
    seq_length = int(params[5])
    trunc_length = int(params[6])
    batch_size = int(params[7])
    hidden_layers_sizes = [int(i) for i in params[1].strip("[]").split(",")]
    fc_layer_size = int(params[2])
    num_biquads = int(params[3])

    impl_ta64 = "torchaudio"
    dtype_ta64 = torch.float64
    impl_tscipy = "scipy"
    dtype_tscipy = torch.float64
    if params[0] == "MODEL1":
        model_ta64 = models.MODEL1(hidden_layers_sizes, num_biquads,
                                   fft_length, impl=impl_ta64, dtype=dtype_ta64)
        model_tscipy = models.MODEL1(hidden_layers_sizes, num_biquads,
                                     fft_length, impl=impl_tscipy, dtype=dtype_tscipy)
        model_tf = models_tf.MODEL1(
            hidden_layers_sizes, num_biquads, fft_length)
    else:
        model_ta64 = models.MODEL2(hidden_layers_sizes, fc_layer_size,
                                   num_biquads, fft_length, impl=impl_ta64, dtype=dtype_ta64)
        model_tscipy = models.MODEL2(hidden_layers_sizes, fc_layer_size,
                                     num_biquads, fft_length, impl=impl_tscipy, dtype=dtype_tscipy)
        model_tf = models_tf.MODEL2(
            hidden_layers_sizes, fc_layer_size, num_biquads, fft_length)
    # select proper device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_weights = torch.load(CURRENT_PATH + "/results/" +
                               directory + "/model.pth", map_location=device)
    model_ta64.load_state_dict(torch_weights)
    model_tscipy.load_state_dict(torch_weights)
    # init with dummy input so that it properly builds the model weights
    model_tf(tf.zeros((1, 32, 1)))
    copy_weights(model_ta64, model_tf)

    # get sample input
    test_input, fs = torchaudio.load(
        CURRENT_PATH + "/data/test/" + data + "-input.wav")
    print("Preprocessing audio")
    start = time.time()
    test_dataset = PreProcess(
        test_input.float(), test_input.float(), seq_length, trunc_length, batch_size
    )
    test_x, _ = next(iter(test_dataset))
    random_idx = np.random.randint(0, test_x.shape[0])
    sample_input = (test_x[random_idx].view(1, -1, 1),)

    # validate implementation
    model_ta64.eval()
    output_ta64 = model_ta64(*sample_input).detach().numpy()
    model_tscipy.eval()
    output_tscipy = model_tscipy(*sample_input).detach().numpy()
    output_tf = model_tf(sample_input[0].numpy())

    print(f"Time elapsed: {time.time() - start:3.1f}s")
    print("- torchaudio64 =? torch_scipy")
    if (np.allclose(output_ta64, output_tscipy, atol=5e-5, rtol=5e-5)):
        print("  ✓ SUCCESS")
    else:
        print("  x FAILURE")

    print("- torchaudio64 =? tf")
    if (np.allclose(output_ta64, output_tf, atol=5e-5, rtol=5e-5)):
        print("  ✓ SUCCESS")
    else:
        print("  x FAILURE")


if __name__ == "__main__":
    main()
