import os
import shutil
import time
import tensorflow as tf
import torchaudio
import auraloss
from preprocess_tf import PreProcess
from tensorflow.summary import create_file_writer
from tensorflow.keras.callbacks import CallbackList, ReduceLROnPlateau, EarlyStopping
from models_tf import MODEL2, MODEL1


def main(
    retrain=False,
    n_epochs=3,
    mr_stft=False,
    model_train="MODEL1",
    data="facebender-rndamp",
    directory="facebender-rndamp_small_MODEL1",
    glu_mlp_hidden_layer_sizes=[3, 4, 5],
    fc_layer_size=5,
    num_biquads=5,
    fir_length=32768,
    seq_length=1024,
    trunc_length=1 * 32768 - 1024,
    overwrite=False,
    batch_size=50,
    learning_rate=1e-2
):

    # neural network architecture:
    # GLU MLP - biquad filters - GLU MLP

    device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")
    print("using", device)

    # MODIFIABLE
    # ------------------------------------------------
    # loss functions
    loss_func = tf.keras.losses.MeanSquaredError()
    loss_func2 = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 512, 256],
        hop_sizes=[120, 50, 25],
        win_lengths=[600, 240, 100],
        mag_distance="L1",
    )
    alpha = 0.001
    scheduler_patience = int(10)  # 10 epochs
    earlystopper_patience = int(n_epochs * 0.1)  # 10% of total number of epochs
    # ------------------------------------------------

    # create folder
    if not retrain:
        if overwrite:
            shutil.rmtree("results/" + directory, ignore_errors=True)
        os.mkdir("results/" + directory)

    # create parameters file
    with open("results/" + directory + "/parameters.txt", "w", encoding="utf-8") as f:
        f.write(
            f"data: {data}\n"
            + f"model_train: {model_train}\n"
            + "layers: "
            + str(glu_mlp_hidden_layer_sizes).replace(" ", "")
            + f"\nlayer: {fc_layer_size}\n"
            + f"num_biquads: {num_biquads}\n"
            + f"fir_length: {fir_length}\n"
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
    writer = create_file_writer("results/" + directory + "/" + "model_" + data)

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
        model = MODEL1(glu_mlp_hidden_layer_sizes, num_biquads, fir_length)
    if model_train == "MODEL2":
        model = MODEL2(
            glu_mlp_hidden_layer_sizes, fc_layer_size, num_biquads, fir_length
        )
    model_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
    )
    if retrain:
        model.load_weights("results/" + directory + "/model_weigths.h5")
        model_optimizer.load_weights(
            "results/" + directory + "/model_optimizer_weigths.h5"
        )

    # callbacks
    callbacks = None
    if scheduler_patience:
        if earlystopper_patience:
            callbacks = CallbackList(
                [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=scheduler_patience, min_lr=1e-6, verbose=1),
                 EarlyStopping(monitor='val_loss', patience=earlystopper_patience, verbose=1)],
                add_history=True, model=model)
        else:
            callbacks = CallbackList(
                [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=scheduler_patience, min_lr=1e-6,
                                   verbose=1)],
                add_history=True, model=model)
    elif earlystopper_patience:
        callbacks = CallbackList([EarlyStopping(monitor='val_loss', patience=earlystopper_patience, verbose=1)],
                                 add_history=True, model=model)
    callback_log = {}  # log to update the monitored values during training (need to update manually!!)
    if callbacks:
        callbacks.on_train_begin(logs=callback_log)

    print("Starting training")

    if retrain:
        best_loss, callback_log = val_loop(
            device,
            model,
            val_dataset,
            mr_stft,
            loss_func,
            loss_func2,
            alpha,
            trunc_length,
            callbacks.callbacks,
            callback_log,
        )
    else:
        best_loss = float("inf")

    start = time.time()
    for epoch in range(n_epochs):
        start_epoch = time.time()
        print(f"Epoch {epoch+1}/{n_epochs}")
        if callbacks:
            callbacks.on_epoch_begin(epoch, logs=callback_log)
        # train for one epoch
        train_loss, callback_log = train_loop(
            device,
            model,
            train_dataset,
            mr_stft,
            loss_func,
            loss_func2,
            alpha,
            trunc_length,
            model_optimizer,
            callbacks,
            callback_log,
        )

        # validation
        if epoch % 2 == 0:
            with tf.GradientTape() as tape:
                val_loss, callback_log = val_loop(
                    device,
                    model,
                    val_dataset,
                    mr_stft,
                    loss_func,
                    loss_func2,
                    alpha,
                    trunc_length,
                    callbacks,
                    callback_log,
                )
            callback_log['val_loss'] = val_loss  # update callback log
            if val_loss < best_loss:
                best_loss = val_loss
                model.save_weights("results/" + directory + "/model_weigths.h5")
                model.save("results/" + directory + "/model.h5")
                model_optimizer.save_weights(
                    "results/" + directory + "/model_optimizer_weigths.h5"
                )
                print("-- New best val loss")
            print(f"-- Train Loss {train_loss:.3E} Val Loss {val_loss:.3E}")
        else:
            print(f"-- Train Loss {train_loss:.3E}")

        # log loss
        with writer.as_default():
            tf.summary.scalar("Train loss", train_loss, step=epoch)
            tf.summary.scalar("Val loss", val_loss, step=epoch)
        end_epoch = time.time()
        print(f"-- Epoch time elapsed: {end_epoch - start_epoch:3.1f}s")

        if callbacks:
            callbacks.on_epoch_end(epoch, logs=callback_log)
        # check if early stopping has triggered stop_training
        if model.stop_training:
            break

    if callbacks:
        callbacks.on_train_end(logs=callback_log)

    writer.flush()
    print(f"Total time elapsed: {time.time() - start:3.1f}s")

    # save final model for retrain
    model.save_weights("results/" + directory + "/model_final.h5")
    model_optimizer.save_weights("results/" + directory + "/model_optimizer_final.h5")


def train_loop(
    device,
    model,
    dataset,
    mr_stft,
    loss_func,
    loss_func2,
    alpha,
    trunc_length,
    model_optimizer,
    callbacks,
    callback_log,
):
    """Train loop for one epoch"""
    train_loss = 0
    batch = 0
    for x, y in dataset:
        if callbacks:
            callbacks.on_batch_begin(batch, logs=callback_log)
            callbacks.on_train_batch_begin(batch, logs=callback_log)

        #x_in = x.to(device)
        #y_out = y.to(device)
        # reset gradient
        #model_optimizer.zero_grad()

        # compute prediction
        with tf.GradientTape() as tape:
            y_hat = model.call(x, training=True)

            # calculate loss
            # truncate before to stabilize filters
            if mr_stft:
                loss = loss_func(
                    y_hat[:, trunc_length:, 0], y[:, trunc_length:, 0]
                ) + alpha * loss_func2(
                    y_hat[:, trunc_length:, :].permute(0, 2, 1),
                    y[:, trunc_length:, :].permute(0, 2, 1),
                )
            else:
                loss = loss_func(y_hat[:, trunc_length:, 0], y[:, trunc_length:, 0])

        # backpropagation
        gradients = tape.gradient(loss, model.trainable_variables)
        model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # accumulate loss
        train_loss += loss.numpy()

        if callbacks:
            callbacks.on_batch_end(batch, logs=callback_log)
            callbacks.on_train_batch_end(batch, logs=callback_log)
        batch += 1

    # return average loss of one epoch
    return train_loss / batch, callback_log


def val_loop(
    device, model, dataset, mr_stft, loss_func, loss_func2, alpha, trunc_length, callbacks, callback_log
):
    """Validation loop for one epoch"""
    val_loss = 0
    batch = 0
    for x, y in dataset:
        if callbacks:
            callbacks.on_batch_begin(batch, logs=callback_log)
            callbacks.on_test_batch_begin(batch, logs=callback_log)

        #x_in = x.to(device)
        #y_out = y.to(device)
        y_hat = model.call(x, training=False)

        if mr_stft:
            loss = loss_func(
                y_hat[:, trunc_length:, 0], y[:, trunc_length:, 0]
            ) + alpha * loss_func2(
                y_hat[:, trunc_length:, :].permute(0, 2, 1),
                y[:, trunc_length:, :].permute(0, 2, 1),
            )
        else:
            loss = loss_func(y_hat[:, trunc_length:, 0], y[:, trunc_length:, 0])

        val_loss += loss.numpy()

        if callbacks:
            callbacks.on_batch_end(batch, logs=callback_log)
            callbacks.on_test_batch_end(batch, logs=callback_log)
        batch += 1

    return val_loss / batch, callback_log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Choose to retrain an existing model or train a new model",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--mr_stft",
        action="store_true",
        default=False,
        help="Use MultiResolutionSTFTLoss",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MODEL1",
        help="Select model (MODEL1 or MODEL2)",
        choices=["MODEL1", "MODEL2"],
    )
    parser.add_argument(
        "--data",
        type=str,
        default="facebender-rndamp",
        help="Select training data (la2a, facebender-rndamp, mcomp-rndamp-A1msR1000ms, ...)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="facebender-rndamp_small_MODEL1_tf",
        help="Select folder name (used in eval.py)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="GLU MLP hidden layer sizes",
    )
    parser.add_argument("--layer", type=int, default=5, help="FC layer size in MODEL2")
    parser.add_argument("--num-biquads", type=int, default=5, help="Number of biquads")
    parser.add_argument(
        "--fir-length",
        type=int,
        default=32768,
        help="Length for estimated FIR filter length",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Samples used for calculating the loss",
    )
    parser.add_argument(
        "--trunc-length",
        type=int,
        default=1 * 32768 - 1024,
        help="Samples used for dividing the audio.\n"
        + "(seq_length and trunc_length should sum to a multiple of fir_length).\n"
        + "(1*fir_length -> no overlap-add method)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing folder, if it exists",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size used to feed the model during training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate used during training",
    )

    args = parser.parse_args()

    main(
        args.retrain,
        args.num_epochs,
        args.mr_stft,
        args.model,
        args.data,
        args.directory,
        args.layers,
        args.layer,
        args.num_biquads,
        args.fir_length,
        args.seq_length,
        args.trunc_length,
        args.overwrite,
        args.batch_size,
        args.learning_rate
    )
