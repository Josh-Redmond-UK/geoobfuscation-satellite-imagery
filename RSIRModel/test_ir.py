import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from src.utils import *


dataset_name = "eurosat"
splits = ["train[:80%]", "train[80%:90%]", "train[90%:]"]
batch_size = 64


# Load the dataset
(train,test,val) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)
train = train.map(lambda x: (x['image'], tf.one_hot(x["label"], num_classes))).batch(batch_size)
test = test.map(lambda x: (x['image'], tf.one_hot(x["label"], num_classes))).batch(batch_size)
val = val.map(lambda x: (x['image'], tf.one_hot(x["label"], num_classes))).batch(batch_size)

feature_extraction = foo
classifier = bar


