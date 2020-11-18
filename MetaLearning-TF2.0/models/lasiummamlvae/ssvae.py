import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import settings
from databases import OmniglotDatabase

import matplotlib.pyplot as plt
import torch

class CheckPointFreq(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, epochs, freq=1, *args, **kwargs):
        super(CheckPointFreq, self).__init__(*args, **kwargs)
        self.freq = freq
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and (epoch + 1) % self.freq == 0:
            super(CheckPointFreq, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self.epochs_since_last_save = np.inf
        self._save_model(self.epochs, logs)

        super(CheckPointFreq, self).on_train_end(logs)


class VisualizationCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, visualization_freq=1, *args, **kwargs):
        super(VisualizationCallback, self).__init__(*args, **kwargs)
        self.visualization_freq = visualization_freq

    def on_epoch_end(self, epoch, logs=None):
        super(VisualizationCallback, self).on_epoch_end(epoch, logs)
        if epoch != 0 and epoch % self.visualization_freq == 0:
            vae = self.model
            for item in vae.get_train_dataset().take(1):
                z_mean, z_log_var, z = vae.encode(item)
                new_item = vae.decode(z)

                # writer = self._get_writer(self._train_run_name)
                # with writer.as_default():
                #     tf.summary.image(name='x', data=item, step=epoch, max_outputs=5)
                #     tf.summary.image(name='x^', data=new_item, step=epoch, max_outputs=5)


class AudioCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, visualization_freq=1, *args, **kwargs):
        super(AudioCallback, self).__init__(*args, **kwargs)
        self.visualization_freq = visualization_freq

    def on_epoch_end(self, epoch, logs=None):
        super(AudioCallback, self).on_epoch_end(epoch, logs)
        if epoch != 0 and epoch % self.visualization_freq == 0:
            vae = self.model
            for item in vae.get_train_dataset().take(1):
                z_mean, z_log_var, z = vae.encode(item)
                new_item = vae.decode(z)

                writer = self._get_writer(self._train_run_name)
                with writer.as_default():
                    tf.summary.audio(name='x', data=item, sample_rate=16000, step=epoch, max_outputs=5)
                    tf.summary.audio(name='x^', data=new_item, step=epoch, sample_rate=16000, max_outputs=5)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SSVAE(keras.Model):
    def __init__(
        self,
        vae_name,
        image_shape,
        latent_dim,
        database,
        parser,
        encoder,
        decoder,
        visualization_freq,
        learning_rate,
        classifier,
        label_dim,
        **kwargs
    ):
        super(SSVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.database = database
        self.parser = parser
        self.visualization_freq = visualization_freq
        self.image_shape = image_shape
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.sampler = Sampling()
        self.vae_name = vae_name
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.y_dim = label_dim
        self.label_dim = label_dim
        self.loss_metric = tf.keras.metrics.Mean()
        self.reconstruction_loss_metric = tf.keras.metrics.Mean()
        self.kl_loss_z_metric = tf.keras.metrics.Mean()
        self.kl_loss_y_metric = tf.keras.metrics.Mean()
        self.kl_loss_metric = tf.keras.metrics.Mean()
        self.ce_loss_metric = tf.keras.metrics.Mean()

    def get_vae_name(self):
        return self.vae_name

    def sample(self, z_mean, z_log_var):
        return self.sampler((z_mean, z_log_var))

    def encode(self, item):
        z_mean, z_log_var = self.encoder(item)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def decode(self, item):
        return self.decoder(item)

    # def duplicate(self, x, y_dim):
    #     return tf.tile(x, tuple(x.shape.as_list()), multiples=y_dim)

    def sample_gaussian(self, m, v):
        eps = tf.random.normal(m)
        z = m + eps * tf.sqrt(v)
        return z

    def kl_cat(self, q, log_q, log_p):
        element_wise = (q * (log_q - log_p))
        kl = tf.reduce_sum(element_wise, axis=-1)
        return kl

    def kl_normal(self, qm, qv, pm = None, pv = None):
        # if pm is None:
        element_wise = 0.5 * (tf.math.log(pv) - tf.math.log(qv) + qv / pv + tf.pow((qm - pm), 2) / pv - 1)
        # else:
        # element_wise = 0.5 * (-tf.math.log(qv) + qv + tf.square(qm) - 1)
        kl = tf.reduce_sum(element_wise, axis=-1)  # element_wise.sum(-1)
        return kl

    def log_bernoulli_with_logits(self, x, logits):
        log_prob = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)
        return log_prob

    def call(self, all_inputs, training=None, mask=None):
        # z_mean, z_log_var = self.encoder(inputs)
        # z = self.sampler([z_mean, z_log_var])
        # reconstruction = self.decoder(z)
        # reconstruction_loss = tf.reduce_mean(
        #     keras.losses.binary_crossentropy(inputs, reconstruction)
        # )
        # reconstruction_loss *= np.prod(self.image_shape)
        # kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        # kl_loss = tf.reduce_mean(kl_loss)
        # kl_loss *= -0.5
        # total_loss = reconstruction_loss + kl_loss # nelbo
        inputs, labelled_dataset = all_inputs
        y_logits = self.classifier(inputs)
        y_logprob = tf.nn.log_softmax(y_logits, axis=1)
        y_prob = tf.nn.softmax(y_logits, axis = 1)
        # print(inputs.shape.as_list()[0], type(inputs.shape.as_list()[0]))
        # print(type(inputs), inputs.shape.as_list())
        # y = np.repeat(np.arange(self.y_dim), 128)
        # inp = np.eye(self.y_dim)[y]
        # y = tf.convert_to_tensor(inp)
        # y = tf.cast(y, dtype = tf.int32)
        # print(y.shape.as_list())

        # x = self.duplicate(inp, self.y_dim)
        # mu, var = self.encoder(inputs, y)  # x, y
        mu, var = self.encoder(inputs)
        # z = self.sample_gaussian(mu, var)
        z = self.sampler([mu, var])
        # decoder_out = self.decoder(z, y)  # z, y
        decoder_out = self.decoder(z)
        kl_y = self.kl_cat(y_prob, y_logprob, np.log(1.0/self.y_dim))

        # kl_z = self.kl_normal(mu, var) # set prior_m = 0, and prior_v = 1
        kl_z = 1 + var - tf.square(mu) - tf.exp(var)
        kl_z = tf.reduce_mean(kl_z)
        kl_z *= -0.5

        rec = tf.reduce_mean(
            keras.losses.binary_crossentropy(inputs, decoder_out)
        )
        rec *= np.prod(self.image_shape)

        kl_y = tf.reduce_mean(kl_y)

        # kl_z = tf.transpose(y_prob, (0, 1)) * kl_z.reshape(self.y_dim, -1)     # y_prob.transpose(0, 1) * kl_z.view(self.y_dim, -1)
        # kl_z = tf.reduce_sum(kl_z, axis=0)          # kl_z.sum(dim=0)

        # rec = tf.transpose(y_prob, (0, 1)) * rec.reshape(self.y_dim, -1)
        # rec = tf.reduce_sum(rec, axis=0)

        nelbo = kl_y + kl_z + rec
        # nelbo = tf.reduce_mean(nelbo)               # nelbo.mean()
        #
        # kl_y = tf.reduce_mean(kl_y)
        # kl_z = tf.reduce_mean(kl_z)
        # rec = tf.reduce_mean(rec)

        reconstruction_loss = rec
        kl_loss = kl_y + kl_z

        subset_input, subset_label = labelled_dataset
        y_logits_subset = self.classifier(subset_input)
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_logits_subset,
                                                          labels = tf.stop_gradient(subset_label))
        ce_total_loss = tf.reduce_mean(ce_loss)
        total_loss = nelbo + ce_total_loss


        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss_z": kl_z,
            "kl_loss_y" : kl_y,
            "kl_loss" : kl_loss,
            "ce_loss" : ce_total_loss
        }

    def test_step(self, data):
        outputs = self.call(data)
        self.loss_metric.update_state(outputs['loss'])
        self.reconstruction_loss_metric.update_state(outputs['reconstruction_loss'])
        self.kl_loss_metric.update_state(outputs['kl_loss'])
        self.ce_loss_metric.update_state(outputs["ce_loss"])
        self.kl_loss_z_metric.update_state(outputs['kl_loss_z'])
        self.kl_loss_y_metric.update_state(outputs['kl_loss_y'])

        return {
            "loss": self.loss_metric.result(),
            "reconstruction_loss": self.reconstruction_loss_metric.result(),
            "ce_loss" : self.ce_loss_metric.result(),
            "kl_loss_z": self.kl_loss_z_metric.result(),
            "kl_loss_y": self.kl_loss_y_metric.result(),
        }

    # TODO

    def train_step(self, data_input):
        data, subset_dataset = data_input

        with tf.GradientTape() as tape:
            outputs = self.call((data, subset_dataset))

        grads = tape.gradient(outputs['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_metric.update_state(outputs['loss'])
        self.reconstruction_loss_metric.update_state(outputs['reconstruction_loss'])
        self.kl_loss_metric.update_state(outputs['kl_loss'])
        self.ce_loss_metric.update_state(outputs["ce_loss"])
        self.kl_loss_z_metric.update_state(outputs['kl_loss_z'])
        self.kl_loss_y_metric.update_state(outputs['kl_loss_y'])

        return {
            "loss": self.loss_metric.result(),
            "reconstruction_loss": self.reconstruction_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result(),
            "ce_loss" : self.ce_loss_metric.result(),
            "kl_loss_z": self.kl_loss_z_metric.result(),
            "kl_loss_y": self.kl_loss_y_metric.result()
        }

    def get_dataset(self, partition='train'):
        instances = self.database.get_all_instances(partition_name=partition)
        random.shuffle(instances)
        train_dataset = tf.data.Dataset.from_tensor_slices(instances).shuffle(len(instances))
        train_dataset = train_dataset.map(self.parser.get_parse_fn())
        train_dataset = train_dataset.batch(128)
        return train_dataset

    def get_dataset_with_labels(self, partition = 'train_labelled_subset'):
        instances = self.database.get_all_instances(partition_name = partition)
        random.shuffle(instances)
        labels = np.zeros((len(instances), self.label_dim))
        for i, path in enumerate(instances):
            image_name = path.split('/')[-1]
            label = image_name.split('_')[0]
            # label = np.random.randint(0, self.label_dim)
            label = int(label) - 1
            labels[i, label] = 1

        train_dataset = tf.data.Dataset.from_tensor_slices(instances)
        train_dataset = train_dataset.map(self.parser.get_parse_fn())
        train_label = tf.data.Dataset.from_tensor_slices(labels)
        train_ds = tf.data.Dataset.zip((train_dataset, train_label)).shuffle(len(instances))
        # elements = []
        # for element in train_ds:
        #     elements.append(element)
        train_ds = train_ds.batch(128)
        return train_ds


    def get_train_dataset(self):
        return self.get_dataset(partition='train')

    def get_val_dataset(self):
        return self.get_dataset(partition='val')

    def get_train_labelled_subset_dataset(self):
        return self.get_dataset_with_labels(partition = 'train_labelled_subset')

    def get_val_labelled_subset_dataset(self):
        return self.get_dataset_with_labels(partition = 'val_labelled_subset')


    def load_latest_checkpoint(self, epoch_to_load_from=None):
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(
                settings.PROJECT_ROOT_ADDRESS,
                'models',
                'lasiummamlssvae',
                'ssvae',
                self.get_vae_name(),
                'ssvae_checkpoints'
            )
        )

        if latest_checkpoint is not None:
            self.load_weights(latest_checkpoint)
            epoch = int(latest_checkpoint[latest_checkpoint.rfind('_') + 1:])
            return epoch

        return -1

    def perform_training(self, epochs, checkpoint_freq=100, vis_callback_cls=None):
        initial_epoch = self.load_latest_checkpoint()
        if initial_epoch != -1:
            print(f'Continue training from epoch {initial_epoch}.')

        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()
        labelled_train_dataset = self.get_train_labelled_subset_dataset()
        labelled_val_dataset = self.get_val_labelled_subset_dataset()
        train_ds = tf.data.Dataset.zip((train_dataset, labelled_train_dataset))
        val_ds = tf.data.Dataset.zip((val_dataset, labelled_val_dataset))
        checkpoint_callback = CheckPointFreq(
            freq=checkpoint_freq,
            filepath=os.path.join(
                settings.PROJECT_ROOT_ADDRESS,
                'models',
                'lasiummamlssvae',
                'ssvae',
                self.get_vae_name(),
                'ssvae_checkpoints',
                'ssvae_{epoch:02d}'
            ),
            save_freq='epoch',
            save_weights_only=True,
            epochs=epochs - 1
        )
        if vis_callback_cls is None:
            vis_callback_cls = VisualizationCallback

        tensorboard_callback = vis_callback_cls(
            log_dir=os.path.join(
                settings.PROJECT_ROOT_ADDRESS,
                'models',
                'lasiummamlssvae',
                'ssvae',
                self.get_vae_name(),
                'ssvae_logs'
            ),
            visualization_freq=self.visualization_freq
        )

        callbacks = [tensorboard_callback, checkpoint_callback]

        self.compile(optimizer=self.optimizer)
        # TODO: test out self.fit
        self.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data= val_ds,
            initial_epoch=initial_epoch
        )

    def visualize_meta_learning_task2(self):
        tf.random.set_seed(10)
        for item in self.get_train_dataset().take(1):
            z_mean, z_log_var, z = self.encode(item)
            fig, axes = plt.subplots(1, 6)
            fig.set_figwidth(6)
            fig.set_figheight(1)

            axes[0].imshow(item[0, ..., 0], cmap='gray')
            for i in range(1, 6):
                axes[i].imshow(self.decode(z + tf.random.normal(shape=z.shape, stddev=0.2 * i))[0, ..., 0], cmap='gray')
                axes[i].set_xlabel(f'noise stddev: {0.2 * i:0.2f}', size='xx-small')

            plt.show()

    def visualize_meta_learning_task(self):
        tf.random.set_seed(10)
        for item in self.get_train_dataset().take(1):
            z_mean, z_log_var, z = self.encode(item)
            new_item = self.decode(z)

            std = tf.exp(0.5 * z_log_var)
            std = 1 / tf.nn.softmax(std) * std

            new_zs = list()
            length = 15
            for i in range(length):
                new_z = z_mean + i / 5 * std
                new_z = new_z[0, ...][tf.newaxis, ...]
                new_zs.append(new_z)

            for i in range(length):
                new_z = z_mean - i / 5 * std
                new_z = new_z[0, ...][tf.newaxis, ...]
                new_zs.append(new_z)

            fig, axes = plt.subplots(length + 1, 2)
            fig.set_figwidth(2)
            fig.set_figheight(length + 1)

            axes[0, 0].imshow(item[0, ..., 0], cmap='gray')
            axes[0, 0].set_xlabel('Real image', size='xx-small')
            axes[0, 1].imshow(new_item[0, ..., 0], cmap='gray')
            axes[0, 1].set_xlabel('Reconstruction', size='xx-small')
            for i in range(1, length + 1):
                new_item = self.decode(new_zs[i - 1][tf.newaxis, ...])
                axes[i, 0].imshow(new_item[0, ..., 0], cmap='gray')
                axes[i, 0].set_xlabel(f'mean + {i / 5} * std', size='xx-small')

                new_item = self.decode(new_zs[length + i - 1][tf.newaxis, ...])
                axes[i, 1].imshow(new_item[0, ..., 0], cmap='gray')
                axes[i, 1].set_xlabel(f'mean - {i / 5} * std', size='xx-small')

            plt.show()

        tf.random.set_seed(None)
