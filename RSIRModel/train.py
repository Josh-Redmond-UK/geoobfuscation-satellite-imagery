import tensorflow_datasets as tfds
import tensorflow as tf
from src.utils import *
import datetime
import pickle

if __name__ == "__main__":
    epochs = 10
    imageSize = 64
    numChannels = 3
    num_classes = 10

    print("loading dataset...")
    # load the whole dataset, for data info
    dataset_name = "eurosat"
    splits = ["train[:80%]", "train[80%:90%]", "train[90%:]"]
    batch_size = 64


    # Load the dataset
    (train,test,val) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)
    train = train.map(lambda x: (x['image'], tf.one_hot(x["label"], num_classes))).batch(batch_size)
    test = test.map(lambda x: (x['image'], tf.one_hot(x["label"], num_classes))).batch(batch_size)
    val = val.map(lambda x: (x['image'], tf.one_hot(x["label"], num_classes))).batch(batch_size)
    print(next(iter(train)))
    input_shape = [None, imageSize, imageSize, numChannels]
    print("loading model...")
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

    # download & load the layer as a feature vector
    keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=True)

    m = tf.keras.Sequential([
keras_layer,
tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    # build the model with input image shape as (64, 64, 3)
    m.build(input_shape)
    m.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    trainin_run_name = f"models/rsir/training_run{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    model_path = os.path.join(trainin_run_name)

    #model_name = "satellite-classification"
    #model_path = os.path.join("models", model_name + ".h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=1)


    #model, checkpoint = getModel()

    # number of training steps
#   n_training_steps   = int(num_examples * 0.6) // batch_size
    # number of validation steps
#   n_validation_steps = int(num_examples * 0.2) // batch_size

    history = m.fit(
        train, validation_data=test,
        verbose=1, epochs=epochs, 
        callbacks=[model_checkpoint]
    )

    with open(trainin_run_name+"/history.pickle", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

