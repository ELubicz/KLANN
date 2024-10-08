import os
import shutil
import time
import tensorflow as tf
import soundfile as sf
import auraloss
from preprocess_tf import PreProcess
from tensorflow.summary import create_file_writer
from tensorflow.keras.callbacks import CallbackList, ReduceLROnPlateau, EarlyStopping
from models_tf import MODEL2, MODEL1
from tqdm import tqdm


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
    learning_rate=1e-2,
):

    # neural network architecture:
    # GLU MLP - biquad filters - GLU MLP

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print("Using", device)
    # Uncomment below for debugging the tensor placements
    #tf.debugging.set_log_device_placement(True)

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
        results_dir = os.path.join("results", directory)
        if overwrite:
            shutil.rmtree(results_dir, ignore_errors=True)
        elif os.path.exists(results_dir):
            raise FileExistsError(
                f"Folder {results_dir} already exists. Use --overwrite to overwrite it."
            )
        os.mkdir("results/" + directory)

    # create parameters file
    with open(os.path.join(results_dir, "parameters.txt"), "w", encoding="utf-8") as f:
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
    train_input, _ = sf.read("data/train/" + data + "-input.wav", dtype="float32")
    train_target, _ = sf.read("data/train/" + data + "-target.wav")
    train_input = tf.convert_to_tensor(train_input, dtype=tf.float32)
    train_target = tf.convert_to_tensor(train_target, dtype=tf.float32)
    val_input, _ = sf.read("data/val/" + data + "-input.wav")
    val_target, _ = sf.read("data/val/" + data + "-target.wav")
    val_input = tf.convert_to_tensor(val_input, dtype=tf.float32)
    val_target = tf.convert_to_tensor(val_target, dtype=tf.float32)
    # DataLoader
    print("Preprocessing audio (train)")
    start = time.time()
    train_dataset = PreProcess(
        train_input, train_target, seq_length, trunc_length, batch_size
    )
    print(f"Time elapsed: {time.time() - start:3.1f}s")
    print("Preprocessing audio (val)")
    start = time.time()
    val_dataset = PreProcess(
        val_input, val_target, seq_length, trunc_length, batch_size
    )
    print(f"Time elapsed: {time.time() - start:3.1f}s")

    model_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
    )
    # initialize model
    print("Initializing model")
    if model_train == "MODEL1":
        model = MODEL1(glu_mlp_hidden_layer_sizes, num_biquads, fir_length, model_optimizer)
    if model_train == "MODEL2":
        model = MODEL2(
            glu_mlp_hidden_layer_sizes, fc_layer_size, num_biquads, fir_length, model_optimizer
        )

    # initialize checkpoints
    checkpoint = tf.train.Checkpoint(optimizer=model_optimizer,
                                     model=model)
    chkpt_manager = tf.train.CheckpointManager(checkpoint, "results/" + directory + "./tf_ckpts", max_to_keep=1)
    if retrain:
        chkpt_manager.restore_or_initialize()
        # reset the learning rate to the initial value
        model_optimizer.learning_rate.assign(learning_rate)
        model.optimizer = model_optimizer

    # callbacks
    callbacks = None
    if scheduler_patience:
        if earlystopper_patience:
            callbacks = CallbackList(
                [
                    ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.1,
                        patience=scheduler_patience,
                        min_lr=1e-6,
                        verbose=1,
                    ),
                    EarlyStopping(
                        monitor="val_loss", patience=earlystopper_patience, verbose=1
                    ),
                ],
                add_history=True,
                model=model,
            )
        else:
            callbacks = CallbackList(
                [
                    ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.1,
                        patience=scheduler_patience,
                        min_lr=1e-6,
                        verbose=1,
                    )
                ],
                add_history=True,
                model=model,
            )
    elif earlystopper_patience:
        callbacks = CallbackList(
            [
                EarlyStopping(
                    monitor="val_loss", patience=earlystopper_patience, verbose=1
                )
            ],
            add_history=True,
            model=model,
        )
    callback_log = (
        {}
    )  # log to update the monitored values during training (need to update manually!!)
    if callbacks:
        callbacks.on_train_begin(logs=callback_log)

    if retrain:
        best_loss, callback_log = val_loop(
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
        print("Training...")
        train_loss, callback_log = train_loop(
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
            print("Validation...")
            val_loss, callback_log = val_loop(
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
            callback_log["val_loss"] = val_loss  # update callback log
            if val_loss < best_loss:
                best_loss = val_loss
                chkpt_manager.save()
                tf.saved_model.save(model, "results/" + directory + "/saved_model")
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
            model_optimizer.learning_rate.assign(model.optimizer.learning_rate)  # refresh the lr of model_optimizer
        # check if early stopping has triggered stop_training
        if model.stop_training:
            break

    if callbacks:
        callbacks.on_train_end(logs=callback_log)

    writer.flush()
    print(f"Total time elapsed: {time.time() - start:3.1f}s")

    # save final model for retrain
    chkpt_manager.save()
    tf.saved_model.save(model, "results/" + directory + "/saved_model_final")


def train_loop(
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
    for x, y in tqdm(dataset):
        if callbacks:
            callbacks.on_batch_begin(batch, logs=callback_log)
            callbacks.on_train_batch_begin(batch, logs=callback_log)

        # compute prediction
        with tf.GradientTape() as tape:
            # add one dimension to the input and output
            x = tf.expand_dims(x, axis=-1)
            y = tf.expand_dims(y, axis=-1)
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
    model,
    dataset,
    mr_stft,
    loss_func,
    loss_func2,
    alpha,
    trunc_length,
    callbacks,
    callback_log,
):
    """Validation loop for one epoch"""
    val_loss = 0
    batch = 0
    for x, y in tqdm(dataset):
        if callbacks:
            callbacks.on_batch_begin(batch, logs=callback_log)
            callbacks.on_test_batch_begin(batch, logs=callback_log)

        # add one dimension to the input and output
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
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
        args.learning_rate,
    )
