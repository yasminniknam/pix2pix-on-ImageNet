import numpy as np
import tensorflow as tf

def convert(img_file):
  return img_file.numpy()

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  # print("**", image_file)
  # print("**", tf.py_function(numpy, image_file).decode('utf-8'))
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image, channels=3)

  
  input_image = tf.cast(image, tf.float32)
  real_image = input_image
   
  image_name = tf.py_function(convert, inp=[image_file])  
  # image_name = image_file.numpy()

  print(image_name)
  type(image_name)
  print('^^^^')
  # image_name = tf.py_function(numpy, image_file).decode('utf-8')
  # image_name = image_name.split('/')[-1][0:-4]
  return input_image, real_image, image_name

OUTPUT_CHANNEL=3
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):
  IMG_WIDTH = 256
  IMG_HEIGHT = 256
  # print(input_image.shape, '&&&&&&&&')

  try:
    # print('lkbkjhbkhjbkjhbkjhbkjhb')
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  except:
    # print('SDFJLB)GSJBDFJBDFJ)LSG')
    input_image = tf.image.grayscale_to_rgb(input_image)
    real_image = tf.image.grayscale_to_rgb(real_image)
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


def occlude(image):
  occlusion_size = np.random.randint(30, 50)
  mask = np.zeros((occlusion_size, occlusion_size, 3))
  random = np.random.randint(0, 200)
  image[random:occlusion_size + random, random:occlusion_size + random, :] = mask
  return image


@tf.function()
def random_jitter(input_image, real_image):

  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # print('###' , tf.shape(input_image))
  # print('###' , input_image.shape)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


def load_image_train(image_file):

  input_image, real_image, _ = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load_image_test(image_file):
  IMG_WIDTH = 256
  IMG_HEIGHT = 256
  input_image, real_image, image_name = load(image_file)
  print(input_image.shape)
  # input_image = occlude(input_image)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  print(input_image.shape)
  input_image, real_image = normalize(input_image, real_image)
  print(input_image.shape)
  print('***')

  return input_image, real_image, image_name

