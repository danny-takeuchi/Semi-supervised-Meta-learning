import random

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from models.maml.maml import ModelAgnosticMetaLearningModel


class MAML_VAE(ModelAgnosticMetaLearningModel):
    def __init__(self, vae, latent_algorithm, *args, **kwargs):
        super(MAML_VAE, self).__init__(*args, **kwargs)
        self.vae = vae
        self.latent_algorithm = latent_algorithm

    def get_config_str(self):
        config_str = super(MAML_VAE, self).get_config_str()
        config_str += f'_{self.latent_algorithm}'
        return config_str

    def get_network_name(self):
        return self.model.name

    def get_parse_function(self):
        return self.vae.parser.get_parse_fn()

    def visualize_meta_learning_task(self, shape, num_tasks_to_visualize=1):
        import matplotlib.pyplot as plt

        dataset = self.get_train_dataset()
        for item in dataset.take(num_tasks_to_visualize):
            fig, axes = plt.subplots(self.k_ml + self.k_val_ml, self.n)
            fig.set_figwidth(self.k_ml + self.k_val_ml)
            fig.set_figheight(self.n)

            (train_ds, val_ds), (_, _) = item
            # Get the first meta batch
            train_ds = train_ds[0, ...]
            val_ds = val_ds[0, ...]
            if shape[2] == 1:
                train_ds = train_ds[..., 0]
                val_ds = val_ds[..., 0]

            for n in range(self.n):
                for k in range(self.k_ml):
                    axes[k, n].imshow(train_ds[n, k, ...])

                for k in range(self.k_val_ml):
                    axes[k + self.k_ml, n].imshow(val_ds[n, k, ...])

            plt.show()

    def generate_with_p3(self, z, z_mean, z_log_var, rotation_index):
        if (rotation_index + 1) % 5 == 0:
            return z + tf.random.normal(shape=z.shape, mean=0, stddev=1.0)

        z = self.vae.sample(z_mean, z_log_var)
        new_z = tf.stack(
            [
                z[0, ...] + (z[(rotation_index + 1) % 5, ...] - z[0, ...]) * 0.4,
                z[1, ...] + (z[(rotation_index + 2) % 5, ...] - z[1, ...]) * 0.4,
                z[2, ...] + (z[(rotation_index + 3) % 5, ...] - z[2, ...]) * 0.4,
                z[3, ...] + (z[(rotation_index + 4) % 5, ...] - z[3, ...]) * 0.4,
                z[4, ...] + (z[(rotation_index + 0) % 5, ...] - z[4, ...]) * 0.4,
            ],
            axis=0
        )

        return new_z

    def generate_with_p2(self, z, z_mean, z_log_var, rotation_index):
        noise = tf.random.normal(shape=z.shape, mean=0, stddev=1.0)
        return z + (noise - z) * 0.4

    def generate_with_p1(self, z, z_mean, z_log_var, rotation_index):
        return z + tf.random.normal(shape=z.shape, mean=0, stddev=tf.random.uniform(shape=(), minval=0, maxval=2))

    def generate_new_z_from_z_data(self, z, z_mean, z_log_var, rotation_index):
        # return self.vae.sample(z_mean, z_log_var)
        # return z + tf.random.normal(shape=z.shape, mean=0, stddev=0.5)
        if self.latent_algorithm == 'p3':
            return self.generate_with_p3(z, z_mean, z_log_var, rotation_index)
        elif self.latent_algorithm == 'p2':
            return self.generate_with_p2(z, z_mean, z_log_var, rotation_index)
        elif self.latent_algorithm == 'p1':
            return self.generate_with_p1(z, z_mean, z_log_var, rotation_index)

        # return z  # + tf.random.normal(shape=z.shape, mean=0, stddev=1.0)

    def augment(self, images):
        new_images = list()
        for i in range(tf.shape(images)[0]):
            new_image = images[i, ...]
            tx = tf.random.uniform((), -5, 5, dtype=tf.int32)
            ty = tf.random.uniform((), -5, 5, dtype=tf.int32)
            transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
            new_image = tfa.image.transform(new_image, transforms, 'NEAREST')

            # new_image = tf.image.random_crop(new_image, size=(64, 64, 3))
            new_images.append(new_image)

        new_images = tf.stack(new_images, axis=0)

        new_images = tf.image.resize(new_images, size=(84, 84))
        return new_images

    def get_train_dataset(self):
        def generate_new_samples_with_vae(instances):
            # from datetime import datetime
            train_indices = [i // self.k_ml + i % self.k_ml * self.n for i in range(self.n * self.k_ml)]
            val_indices = [
                self.n * self.k_ml + i // self.k_val_ml + i % self.k_val_ml * self.n
                for i in range(self.n * self.k_val_ml)
            ]

            # TODO test speed change with this tf.function and without it.
            # @tf.function
            def f(instances):
                # current_time = datetime.now()
                z_mean, z_log_var, z = self.vae.encode(instances)
                # print(f'encode time spent: {datetime.now() - current_time}')

                # current_time = datetime.now()
                new_zs = list()
                for i in range(self.k_ml + self.k_val_ml - 1):
                    new_z = self.generate_new_z_from_z_data(z, z_mean, z_log_var, rotation_index=i)
                    new_zs.append(new_z)
                new_zs = tf.concat(new_zs, axis=0)
                # print(f'generate z time spent: {datetime.now() - current_time}')

                # current_time = datetime.now()
                new_instances = self.vae.decode(new_zs)
                # print(f'decode time spent: {datetime.now() - current_time}')

                # current_time = datetime.now()
                new_instances = tf.concat((instances, new_instances), axis=0)

                train_instances = tf.gather(new_instances, train_indices, axis=0)
                val_instances = tf.gather(new_instances, val_indices, axis=0)

                val_instances = self.augment(val_instances)

                return (
                    tf.reshape(train_instances, (self.n, self.k_ml, *train_instances.shape[1:])),
                    tf.reshape(val_instances, (self.n, self.k_val_ml, *val_instances.shape[1:])),
                )

            return tf.py_function(f, inp=[instances], Tout=[tf.float32, tf.float32])

        instances = self.database.get_all_instances(partition_name='train')
        random.shuffle(instances)

        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(self.get_parse_function())
        # dataset = dataset.shuffle(buffer_size=len(instances))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.n, drop_remainder=True)

        dataset = dataset.map(generate_new_samples_with_vae)
        labels_dataset = self.data_loader.make_labels_dataset(self.n, self.k_ml, self.k_val_ml, one_hot_labels=True)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(self.meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', tf.data.experimental.cardinality(dataset))
        return dataset




