import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from tensorflow import keras
from tensorflow.keras import layers

from databases import OmniglotDatabase
import tensorflow as tf
from models.lasiummamlvae.database_parsers import OmniglotParser
from models.lasiummamlvae.maml_vae import MAML_VAE
from models.lasiummamlvae.vae import VAE
from networks.maml_umtra_networks import SimpleModel



def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
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


    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    return encoder


def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
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
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.config.experimental_run_functions_eagerly(True)

    omniglot_database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    shape = (28, 28, 1)
    latent_dim = 20
    omniglot_encoder = get_encoder(latent_dim)
    omniglot_decoder = get_decoder(latent_dim)
    omniglot_parser = OmniglotParser(shape=shape)

    print(os.path.dirname(sys.argv[0]))

    vae = VAE(
        'omniglot',
        image_shape=shape,
        latent_dim=latent_dim,
        database=omniglot_database,
        parser=omniglot_parser,
        encoder=omniglot_encoder,
        decoder=omniglot_decoder,
        visualization_freq=5,
        learning_rate=0.001,
    )
    vae.perform_training(epochs=500, checkpoint_freq=50)
    vae.load_latest_checkpoint()
    # vae.visualize_meta_learning_task()

    # simple_model = SimpleModel(num_classes=1683)

    maml_vae = MAML_VAE(
        vae=vae,
        database=omniglot_database,
        latent_algorithm='p1',
        network_cls=SimpleModel,
        n=5,
        k_ml=15,
        k_val_ml=15,
        k_val=15,
        k_val_val=15,
        k_test=15,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=50,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=500,
        clip_gradients=False,
        experiment_name='omniglot_vae_k=15_all_k_v4', # 'omniglot_vae_0.5_shift_run2',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    # maml_vae.visualize_meta_learning_task(shape, num_tasks_to_visualize=2)

    maml_vae.train(iterations=10000)
    maml_vae.evaluate(500, seed=42, num_tasks=1000)
