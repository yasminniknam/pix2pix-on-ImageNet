import os
import time
import datetime
import tensorflow as tf
from IPython import display

from model import *
from functions import *

tf.keras.backend.clear_session()

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


def fit(train_ds, steps, test_ds=None):

    start = time.time()
    # print('****************INJJAAA*****')
    # print(train_ds.repeat().take(steps).enumerate())
    # print('tamum')
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        # print('Step number: ', step)
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            # generate_images(generator, example_input, example_target)
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# The facade training set consist of 400 images
BUFFER_SIZE = 256
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 16
# Each image is 256x256 in size
IMG_WIDTH = 128
IMG_HEIGHT = 128
OUTPUT_CHANNELS = 3

PATH = './train2017'


generator = Generator()
discriminator = Discriminator()

train_dataset = tf.data.Dataset.list_files(PATH + '/*.jpg')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.map(lambda x: tf.py_function(load_image_train, [x], [tf.string]),
#                                   num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = '/home/yasamin/scratch/pix2pix/pix2pix-on-coco/pix2pix-on-ImageNet/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                                 generator=generator, discriminator=discriminator)

log_dir = "/home/yasamin/scratch/pix2pix/pix2pix-on-coco/pix2pix-on-ImageNet/logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

fit(train_dataset, steps=int(118287/BATCH_SIZE)*100)
