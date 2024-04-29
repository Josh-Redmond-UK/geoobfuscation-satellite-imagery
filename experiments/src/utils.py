import PIL
import numpy as np
import tensorflow as tf
import random
import opensimplex


def get_poi(image, num_points, buffer_range):
  buffer_size = random.randint(buffer_range[0], buffer_range[1])

  point_xs = random.sample(range(buffer_size, image.shape[1]), num_points)
  point_ys = random.sample(range(buffer_size, image.shape[0]), num_points)
  imgShape = image.shape
  bool_mask =np.zeros(imgShape)

  image = tf.cast(image, tf.float32)



  for origin in list(zip(point_xs, point_ys)):
    min_x = origin[0]- buffer_size
    max_x = origin[0]+ buffer_size
    min_y = origin[1]- buffer_size
    max_y = origin[1]+ buffer_size

    bool_mask[min_y:max_y, min_x:max_x] = 1
    #print(np.mean(patch))



  new_img = tf.identity(image)
  new_img *= tf.cast(tf.convert_to_tensor(bool_mask), tf.float32)


  return new_img, image

def getW(imClass, rClasses):
  w = []
  for r in rClasses:
    rC = np.argmax(r)
    pqk = np.squeeze(imClass)[rC]
    w.append(1-pqk)

  return np.array(w)

def getEuclidDist(qFeats, rFeats):
  d = np.apply_along_axis(lambda x: abs(np.linalg.norm(qFeats - x)), 1, rFeats)
  return d


def get_wd(q_image, feature_extraction, classifier, r_feats, r_classes):
    q_image = tf.expand_dims(q_image, 0)
    q_feat = feature_extraction.predict(q_image)
    q_class = classifier.predict(q_image)
    return getW(q_class, r_classes) * getEuclidDist(q_feat, r_feats)
   
def paint_poi(target, poi):
   
    target[poi != 0] = 0
    return target + poi


def create_poi_image(image, poi_num, poi_size):
    poi, _image = get_poi(image, poi_num, [poi_size, poi_size])
    
    return poi

def blur(image):
    image = np.array(image)*255
    image = image.astype(np.uint8)
    image = np.clip(image, 0, 255)
    return np.array(PIL.Image.fromarray(image).filter(PIL.ImageFilter.BLUR))/255

def pixelate(image):
    image = np.array(image)*255
    image = image.astype(np.uint8)
    image = np.clip(image, 0, 255)
    return np.array(PIL.Image.fromarray(image).resize((16,16), PIL.Image.NEAREST).resize((64,64), PIL.Image.NEAREST))/255


def create_noise(h, w, seed=2, FEATURE_SIZE=40, freq_sine = 8):
    
    size = w
    j = 0

    opensimplex.seed(seed)

    array = np.zeros([h,w])
    for y in range(h):
        for x in range(w):
            array[x, y] = opensimplex.noise4(x=x/FEATURE_SIZE, y=y/FEATURE_SIZE, z=0.0, w=0.0)
            
    array = np.sin(array * freq_sine * np.pi)
    return array
    
def perturb(image, noise, budget):
    noise = np.sign((noise - 0.5) * 2) * budget
    noise = np.clip(noise, np.maximum(-image, -budget), np.minimum(1 - image, budget))

    return image + noise


def clouds(image):
    ## Anisotropic clouds
    noise = np.expand_dims(create_noise(64,64, freq_sine=24), axis=-1)
    image = np.clip(perturb(image, noise, 0.05), 0, 1)

    image = np.array(image)#*255
    #image = image.astype(np.uint8)
    #image = np.clip(image, 0, 255)
    return image#np.array(PIL.Image.fromarray(image))/255 #fix later




def repaint_from_sample(poi_image, model):
    img_size = poi_image.shape[0]
    num_images = 1
    # 1. Randomly sample noise (starting point for reverse process)
    samples = tf.convert_to_tensor(np.random.normal(size=(num_images, img_size, img_size, 3)), dtype=tf.float32)
    #samples

    noise = samples
    poi_image = tf.expand_dims(poi_image, 0)
    boolMask = poi_image ==0
    boolInvMask = poi_image != 0
    test = (np.array(boolMask)*np.array(samples))+np.array(poi_image)
    sampleNoiseNoCon = noise



    # 2. Sample from the model iteratively
    for t in reversed(range(0, model.timesteps)):
        tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)

        noiseToAdd = model.gdf_util.q_sample(poi_image, t, noise)
        test = (np.array(boolMask)*np.array(samples))+(np.array(noiseToAdd)*np.array(boolInvMask))
        samples = tf.expand_dims(tf.convert_to_tensor(np.squeeze(test)), 0)

        pred_noise = model.ema_network.predict(
            [samples, tt], verbose=0, batch_size=num_images
        )
        samples = model.gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=True
        )

    # 3. Return generated samples
    testInt = np.squeeze((
        tf.clip_by_value(samples * 127.5 + 127.5, 0.0, 255.0)
        .numpy()
        .astype(np.uint8)))

    return testInt/255

def repaint(poi_image, model):
    return repaint_from_sample(poi_image, model)


