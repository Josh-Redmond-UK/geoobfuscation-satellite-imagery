import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000, num_classes=10):
  ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
  # shuffle the dataset
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # split to batches
  ds = ds.batch(batch_size)

  return ds



def getModel(inputShape = [None, 64, 64, 3], num_classes=10):
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

    # download & load the layer as a feature vector
    keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=True)

    m = tf.keras.Sequential([
  keras_layer,
  tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    # build the model with input image shape as (64, 64, 3)
    m.build(inputShape)
    m.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy", "precision", "recall"]
    )

    model_name = "satellite-classification"
    model_path = os.path.join("models", model_name + ".h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=1)

    return m, model_checkpoint

def trainModel(model, epochs, train_ds, valid_ds, n_training_steps, n_validation_steps, model_checkpoint):
    # train the model
    history = model.fit(
        train_ds, validation_data=valid_ds,
        steps_per_epoch=n_training_steps,
        validation_steps=n_validation_steps,
        verbose=1, epochs=epochs, 
        callbacks=[model_checkpoint]
    )
    model.save("models/satellite-classification.h5")
    return history

def prepare_for_test(ds, cache=True, batch_size=64, shuffle_buffer_size=1000, num_classes=10):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
  # shuffle the dataset
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # split to batches
  ds = ds.batch(batch_size)
  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

def getW(imClass, rClasses):
  w = []
  print(imClass)
  for r in rClasses:
    rC = np.argmax(r)
    print(rC, imClass)
    pqk = np.squeeze(imClass)[rC]
    w.append(1-pqk)

  return np.array(w)

def getEuclidDist(qFeats, rFeats):
  d = np.apply_along_axis(lambda x: abs(np.linalg.norm(qFeats - x)), 1, rFeats)
  return d


def lookupIdxs(idxs, dataset):
  return np.take(dataset, idxs, 0)

def lookupImage(q, rClasses, rFeatures, R, k=3):
  idxs = getNearestIdxs(q, rClasses, rFeatures, k)
  return lookupIdxs(idxs, R)

def plotLookup(q, rLookup, figName="results"):
  fig, axs = plt.subplots(rLookup.shape[0]+1, figsize=(10,10))
  axs[0].imshow(q)
  for idx, r in enumerate(rLookup):
    axs[idx+1].imshow(r)
  plt.savefig(f"{figName}.png")

def get_wd(query_set, lookup_set, feature_extraction, classifier):
    q_features = feature_extraction(query_set)
    l_features = feature_extraction(lookup_set)

    q_class = classifier(query_set)
    l_class = classifier(lookup_set)

    wd_list = []

    for q in range(len(query_set)):
        w = getW(q_class[q], l_class)
        d = getEuclidDist(q_features[q], l_features)
        wd = w*d
        wd_list.append(wd)


    return wd_list
    



def getNearestIdxs(image, rClasses, rFeatures, model, featureExtraction, k = 3, returnWD=False):
 
  allClasses = rClasses
  allFeatures = rFeatures

  imgClass = model.predict(image)
  imgFeats = featureExtraction.predict(image)

  w = getW(imgClass, allClasses)
  d = getEuclidDist(imgFeats, allFeatures)

  wd = w*d
  if k >= wd.size:
    k = wd.size-1
  idxs = np.argsort(wd)[::-1][:k]
  
  if not returnWD:
    return idxs
  else:
    return idxs, wd
