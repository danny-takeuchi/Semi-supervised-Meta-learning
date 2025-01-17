import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase
import tensorflow as tf
from models.lasiummamlvae.database_parsers import OmniglotParser
from models.lasiummamlvae.maml_vae import MAML_VAE
# from models.lasiummamlvae.vae import VAE
from models.lasiummamlvae.ssvae import SSVAE
from networks.maml_umtra_networks import SimpleModel
import copy



def get_encoder(latent_dim, label_dim):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    # label_input = keras.Input(shape = (label_dim, ))
    x = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", use_bias=False)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    # x = layers.Concatenate(axis = -1)([x, label_input]) # (batch_size, 784 + 1623)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # encoder = keras.Model([encoder_inputs, label_input], [z_mean, z_log_var], name="encoder")
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    return encoder



def get_decoder(latent_dim, label_dim):
    latent_inputs = keras.Input(shape=(latent_dim,)) # (batch_size, latent_dim)
    label_input = keras.Input(shape = (label_dim, ))
    # x = layers.Concatenate(axis = 1)([latent_inputs, label_input])
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    # x = layers.Dense(300, activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dense(300, activation = 'relu')(x)
    # x = layers.BatchNormalization()(x)
    # decoder_outputs = layers.Dense(784)(x)
    decoder = keras.Model([latent_inputs, label_input], decoder_outputs, name="decoder")
    decoder.summary()

    return decoder

def get_classifier(latent_dim, label_dim):
    classifier_inputs = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", use_bias=False)(classifier_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    #
    # x = layers.Dense(latent_dim, activation = 'relu')(x)
    # x = layers.Dense(latent_dim, activation = 'relu')(x)
    output = layers.Dense(label_dim)(x)
    classifier = keras.Model(classifier_inputs, output, name="classifier")
    classifier.summary()
    return classifier

if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    omniglot_database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    shape = (28, 28, 1)
    latent_dim = 20
    label_dim = 1683
    omniglot_encoder = get_encoder(latent_dim, label_dim)
    omniglot_decoder = get_decoder(latent_dim, label_dim)
    omniglot_classifier = get_classifier(latent_dim, label_dim)
    omniglot_parser = OmniglotParser(shape=shape)

    vae = SSVAE(
        'omniglot',
        image_shape=shape,
        latent_dim=latent_dim,
        database=omniglot_database,
        parser=omniglot_parser,
        encoder=omniglot_encoder,
        decoder=omniglot_decoder,
        visualization_freq=5,
        classifier = omniglot_classifier,
        learning_rate=0.001,
        label_dim = label_dim
    )
    vae.perform_training(epochs=500, checkpoint_freq=100)
    vae.load_latest_checkpoint()
    vae.visualize_meta_learning_task()

    maml_vae = MAML_VAE(
        vae=vae,
        database=omniglot_database,
        latent_algorithm='p1',
        network_cls=SimpleModel,
        n=5,
<<<<<<< HEAD
        k_ml=10, #1
        k_val_ml=10,
        k_val=10, #1
=======
        k_ml=10,
        k_val_ml=10,
        k_val=10,
>>>>>>> b549d6bb67c87d9d07b1d4519d447becaa4e265a
        k_val_val=10,
        k_test=10,
        k_val_test=10,
        meta_batch_size=4,
        num_steps_ml=1,
        lr_inner_ml=0.4,
        num_steps_validation=50,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
<<<<<<< HEAD
        num_tasks_val=100,
        clip_gradients=False,
=======
        num_tasks_val=500,
        clip_gradients=True,
>>>>>>> b549d6bb67c87d9d07b1d4519d447becaa4e265a
        experiment_name='omniglot_ssvae_k=1_all_k',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    maml_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    maml_vae.train(iterations=10000)
<<<<<<< HEAD
    maml_vae.evaluate(50, seed=42, num_tasks=1000)
=======
    maml_vae.evaluate(500, seed=42, num_tasks=1000)
>>>>>>> b549d6bb67c87d9d07b1d4519d447becaa4e265a
