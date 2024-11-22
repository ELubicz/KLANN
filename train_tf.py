import os
import shutil
import time
import datetime
import keras
import tensorflow as tf
import soundfile as sf
import auraloss
from pathlib import Path
from preprocess_tf import PreProcess
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
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
    loss_func1 = tf.keras.losses.MeanSquaredError()
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

    # define custom loss function
    def loss_computation(y_true, y_pred, func1, func2, multires_stft, trunc_len, alpha_param):
        # calculate loss
        # truncate before to stabilize filters
        if multires_stft:
            loss = func1(
                y_pred[:, trunc_len:, 0], y_true[:, trunc_len:, 0]
            ) + alpha_param * func2(
                y_pred[:, trunc_len:, :].permute(0, 2, 1),
                y_true[:, trunc_len:, :].permute(0, 2, 1),
            )
        else:
            loss = func1(y_pred[:, trunc_len:, 0], y_true[:, trunc_len:, 0])

        return loss
    # model.fit() function requires a loss function with two arguments: (y_true, y_pred)
    def model_loss(func1, func2, multires_stft, trunc_len, alpha_param):
        def loss(y_true, y_pred):
            return loss_computation(y_true, y_pred, func1, func2, multires_stft, trunc_len, alpha_param)
        return loss
    # now we can initialize the loss function
    loss_func = model_loss(loss_func1, loss_func2, mr_stft, trunc_length, alpha)

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
    input_shape = [seq_length + trunc_length, 1]
    if retrain:
        model = keras.models.load_model("results/" + directory + "/saved_model_final.keras")
    else:
        if model_train == "MODEL1":
            model_inst = MODEL1(glu_mlp_hidden_layer_sizes, num_biquads, fir_length, model_optimizer)
        if model_train == "MODEL2":
            model_inst = MODEL2(
                glu_mlp_hidden_layer_sizes, fc_layer_size, num_biquads, fir_length, model_optimizer
            )
        model = model_inst.get_model(input_shape, training=True)
    
        # compile model
        model.compile(optimizer=model_optimizer, loss=loss_func)
    model.summary()

    # callbacks
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = Path(__file__).parent / "tb_logs" / (time_stamp + "_" + directory)
    callbacks = None
    if scheduler_patience:
        if earlystopper_patience:
            callbacks = [
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
                TensorBoard(
                    log_dir=tb_log_dir, 
                    histogram_freq=0
                ),
                ModelCheckpoint(
                    filepath="results/" + directory + "/saved_model_best.keras",
                    save_best_only=True, 
                    monitor='val_loss', 
                    mode='min',
                    save_weights_only=False
                )
            ]
        else:
            callbacks = [
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    patience=scheduler_patience,
                    min_lr=1e-6,
                    verbose=1,
                ),
                TensorBoard(
                    log_dir=tb_log_dir, 
                    histogram_freq=0
                ),
                ModelCheckpoint(
                    filepath="results/" + directory + "/saved_model_best.keras",
                    save_best_only=True, 
                    monitor='val_loss', 
                    mode='min',
                    save_weights_only=False
                )
            ]
    elif earlystopper_patience:
        callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=earlystopper_patience, verbose=1
                ),
                TensorBoard(
                    log_dir=tb_log_dir, 
                    histogram_freq=0
                ),
                ModelCheckpoint(
                    filepath="results/" + directory + "/saved_model_best.keras",
                    save_best_only=True, 
                    monitor='val_loss', 
                    mode='min',
                    save_weights_only=False
                )
            ]
    
    # train
    start = time.time()
    print("Training...")
    hist = model.fit(
        train_dataset,
        epochs=n_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        validation_freq=2
    )
    print(f"Total time elapsed: {time.time() - start:3.1f}s")

    # save final model for retrain
    tf.saved_model.save(model, "results/" + directory + "/saved_model_final.keras")

#x = tf.expand_dims(x, axis=-1)
#y = tf.expand_dims(y, axis=-1)


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
