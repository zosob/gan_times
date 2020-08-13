from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD

import matplotlib.pyplot as plt

import sys
import time
import numpy as np


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = SGD(0.01)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()
        lst_loss=[]
        lst_dloss = []
        lst_acc= []

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            lst_loss.append(g_loss)
            lst_dloss.append(d_loss[0])
            lst_acc.append(100*d_loss[1])
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch,batch_size)

        #draw_result(range(epochs), lst_loss, lst_dloss, lst_acc, "sgd_method",batch_size)
        plt.figure(batch_size)
        plt.plot(range(epochs), lst_dloss, '-b', label='loss discriminator')
        plt.plot(range(epochs), lst_acc, '-r', label='accuracy')
        plt.plot(range(epochs), lst_loss, '-g', label='loss generator')

        plt.xlabel("Iterations")
        plt.legend(loc='upper left')
        plt.title("SGD Method")

        # save image
        plt.savefig("title%d.png" %batch_size)  # should before show method

        # show
        #plt.show()


    #def draw_result(lst_iter, lst_gloss, lst_dloss, lst_acc, title,batch_size):

    #    plt.plot(lst_iter, lst_gloss, '-g', label='loss generator')

#        plt.xlabel("Iterations")
#
        # save image
#        plt.savefig("title%d.png" %batch_size)  # should before show method

        # show
#        plt.show()


    def sample_images(self, epoch, batch_size):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images%d/%d.png" % (batch_size,epoch))
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    begin8=time.time()
    gan.train(epochs=30000, batch_size=8, sample_interval=200)
    end8=time.time()
    begin16=time.time()
    gan.train(epochs=30000, batch_size=16, sample_interval=200)
    end16=time.time()
    begin32=time.time()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)
    end32=time.time()
    begin50=time.time()
    gan.train(epochs=30000, batch_size=50, sample_interval=200)
    end50=time.time()
    begin64=time.time()
    gan.train(epochs=30000, batch_size=64, sample_interval=200)
    end64=time.time()
    begin100= time.time()
    gan.train(epochs=30000, batch_size=100, sample_interval=200)
    end100=time.time()
    begin128=time.time()
    gan.train(epochs=30000, batch_size=128, sample_interval=200)
    end128=time.time()
    begin150=time.time()
    gan.train(epochs=30000, batch_size=150, sample_interval=200)
    end150=time.time()
    begin200=time.time()
    gan.train(epochs=30000, batch_size=200, sample_interval=200)
    end200=time.time()
    begin250=time.time()
    gan.train(epochs=30000, batch_size=250, sample_interval=200)
    end250=time.time()
    begin256=time.time()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)
    end256=time.time()
    begin512= time.time()
    gan.train(epochs=30000, batch_size=512, sample_interval=200)
    end512=time.time()
    begin1024= time.time()
    gan.train(epochs=30000, batch_size=1024, sample_interval=200)
    end1024=time.time()
    print("Time for 8 batches:", end8-begin8)
    print("Time for 16 batches:", end16-begin16)
    print("Time for 32 batches:", end32-begin32)
    print("Time for 50 batches:", end50-begin50)
    print("Time for 64 batches:", end64-begin64)
    print("Time for 100 batches:", end100-begin100)
    print("Time for 128 batches:", end128-begin128)
    print("Time for 150 batches:", end150-begin150)
    print("Time for 200 batches:", end200-begin200)
    print("Time for 250 batches:", end250-begin250)
    print("Time for 256 batches:", end256-begin256)
    print("Time for 512 batches:", end512-begin512)
    print("Time for 1024 batches:", end1024-begin1024)
