import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from src.utils import *
from models.diffusion.utils import *
import time
import pickle
import os
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"

num_classes = 10
total_timesteps = 1200
experiment_time = time.strftime("%Y%m%d-%H%M%S")
results_dir = f"generated_images_{experiment_time}"
#tf.config.set_visible_devices([], 'GPU') # Disable GPU
#starting_time_string = 
(train,test,val) = tfds.load("eurosat", split=["train[0%:70%]", "train[70%:85%]", "train[99%:100%]"], with_info=False, shuffle_files=True)
train = train.map(lambda x: x['image']/255)
test = test.map(lambda x: x['image']/255)
val = val.map(lambda x: x['image']/255)


val_np = val.as_numpy_iterator()

## Experiment Parameters
# Blurring
# pixelate
# anisotropic clouds
# repaint
l_set = []
for _v in val_np:
    l_set.append(_v)


print("Loaded dataset")

model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

# download & load the layer as a feature vector
keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=True)

rsir_model = tf.keras.Sequential([
  keras_layer,
  tf.keras.layers.Dense(num_classes, activation="softmax")
])
print("loading weights")
# build the model with input image shape as (64, 64, 3)
rsir_model.build([None, 64, 64, 3])
rsir_model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
rsir_model.load_weights("models/rsir/RSIR_27_02_2024.h5")


feature_extraction = tf.keras.Model(inputs=rsir_model.input,
                                 outputs=rsir_model.layers[0].output)


classifier = rsir_model



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
ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
#ema_network.set_weights(network.get_weights())  # Initially the weights are the same

# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
diffusion_model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# Compile the model
diffusion_model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)


diffusion_model.ema_network.load_weights("models/diffusion/ema_weights.h5")
diffusion_model.network.load_weights("models/diffusion/normal_weights.h5")


print("Models loaded")



print("Loaded dataset")
results = []

val = val.map(lambda x: tf.expand_dims(x, 0))
r_features = feature_extraction.predict(val)
r_classes = classifier.predict(val)   




if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("Extracted features")
print("Starting experiments")
for idx, q in enumerate(l_set):
    start = time.time()

    poi = create_poi_image(q, 3, 10)
    blurred = paint_poi(blur(q) , poi)
    pixelated = paint_poi(pixelate(q) , poi)
    clouded = clouds(q) 
    repainted = repaint(poi , diffusion_model)

    plt.imshow(poi)
    plt.savefig(f"{results_dir}/poi_{idx}.png")
    plt.close()

    plt.imshow(q)
    plt.savefig(f"{results_dir}/original_{idx}.png")
    plt.close()

    plt.imshow(repainted)
    plt.savefig(f"{results_dir}/repainted_{idx}.png")
    plt.close()

    plt.imshow(clouded)
    plt.savefig(f"{results_dir}/clouded_{idx}.png")
    plt.close()

    plt.imshow(pixelated)
    plt.savefig(f"{results_dir}/pixelated_{idx}.png")
    plt.close()

    plt.imshow(blurred)
    plt.savefig(f"{results_dir}/blurred_{idx}.png")
    plt.close()
    

    blurred_wd = get_wd(blurred, feature_extraction, classifier, r_features, r_classes)
    pixelate_wd = get_wd(pixelated, feature_extraction, classifier, r_features, r_classes)
    clouds_wd = get_wd(clouded, feature_extraction, classifier, r_features, r_classes)
    repaint_wd = get_wd(repainted, feature_extraction, classifier, r_features, r_classes)

    results.append({'blurred': blurred_wd, 'pixelate': pixelate_wd, 'clouds': clouds_wd, 'repaint': repaint_wd})
    print(f"Finished {idx} of {len(l_set)} in {time.time()-start} seconds")
    #print(f"Finished experiments in {time.time()-start} seconds") 

pickle.dump(results, open(f"{results_dir}/results.pkl", "wb"))


blurred_wd = []
pixelate_wd = []
clouds_wd = []
repaint_wd = []

for idx, r in enumerate(results):
    pixelate_wd.append(r['pixelate'][idx])
    blurred_wd.append(r['blurred'][idx])
    clouds_wd.append(r['clouds'][idx])
    repaint_wd.append(r['repaint'][idx])

global_min = min(min(blurred_wd), min(pixelate_wd), min(clouds_wd), min(repaint_wd))
global_max = max(max(blurred_wd), max(pixelate_wd), max(clouds_wd), max(repaint_wd))
alpha = 0.25
plt.figure(figsize=(10,10))
plt.hist(blurred_wd, label='blurred', range=(global_min, global_max), alpha=alpha)
plt.hist(pixelate_wd, label='pixelate', range=(global_min, global_max), alpha=alpha )
plt.hist(clouds_wd, label='clouds', range=(global_min, global_max), alpha=alpha)
plt.hist(repaint_wd, label='repaint', range=(global_min, global_max), alpha=alpha)
plt.legend()
plt.savefig(f'{results_dir}/results.png')
plt.close()



