from src.utils import *
import tensorflow_datasets as tfds
import pickle
import datetime

def main():
    dataset_path = ""

    #hyper params
    batch_size = 32
    num_epochs = 50
    total_timesteps = 1000
    norm_groups = 8  # Number of groups used in GroupNormalization layer
    learning_rate = 2e-4

    img_size = 64
    img_channels = 3
    clip_min = -1.0
    clip_max = 1.0

    first_conv_channels = 64
    channel_multiplier = [1, 2, 4, 8]
    widths = [first_conv_channels * mult for mult in channel_multiplier]
    has_attention = [False, False, True, True]
    num_res_blocks = 2  # Number of residual blocks

    dataset_name = "eurosat"
    splits = ["train[:80%]", "train[80%:90%]"]


    # Load the dataset
    (ds,val) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)
    print("Got dataset")

    augmentLayer = augment()



    train_ds = (
        ds.map(lambda x: train_preprocessing(x, img_size, augmentLayer), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .shuffle(batch_size * 2)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val.map(lambda x: test_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .shuffle(batch_size * 2)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build the unet model
    network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )
    # Build the unet model (ema weights)
    ema_network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )

    trainin_run_name = f"models/diffusion/training_run{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"


    emaCallback = save_EMA_weights(filepath=trainin_run_name+'/ema_weights')


    ema_network.set_weights(network.get_weights())  # Initially the weights are the same

    # Get an instance of the Gaussian Diffusion utilities
    gdf_util = GaussianDiffusion(timesteps=total_timesteps)

    # Get the model
    model = DiffusionModel(
        network=network,
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=total_timesteps,
    )


    saveCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = trainin_run_name,
        save_best_only = True
    )

    # Compile the model
    model.compile(
        loss=tf.keras.metrics.mean_squared_error,
        optimizer="adam",

    )



    #Train the model
    history = model.fit(
        train_ds,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images), saveCheckpoint, emaCallback],
        validation_data=val_ds)

    with open(trainin_run_name+"/history.pickle", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return model, history
