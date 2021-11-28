import os
from model import *
from functions import *

tf.keras.backend.clear_session()

BUFFER_SIZE = 256
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 128
IMG_HEIGHT = 128
OUTPUT_CHANNELS = 3

generator = Generator()
discriminator = Discriminator()

PATH = './val2017'
test_dataset = tf.data.Dataset.list_files(PATH + '/*.jpg')
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = '/home/yasamin/scratch/pix2pix/pix2pix-on-coco/pix2pix-on-ImageNet/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                                 generator=generator, discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#addr = './val'
addr = '/home/yasamin/scratch/pix2pix/pix2pix-on-coco/pix2pix-on-ImageNet/results/occluded_model/output_occluded_input'

# Run the trained model on a few examples from the test set
for inp, tar, name in test_dataset:
    img_name = name.numpy()[0]
    img_name = img_name[10:-4]
    img_name = img_name.decode("utf-8")
    print(img_name)
    generate_images(generator, inp, tar, img_name, addr)
