from models.gansampling.gan_sampling import GANSampling
from networks.maml_umtra_networks import MiniImagenetModel
from databases import CelebADatabase


def run_celeba():
    celeba_database = CelebADatabase()

    gan_sampling = GANSampling(
        database=celeba_database,
        network_cls=MiniImagenetModel,
        n=5,
        k=1,
        k_val_ml=6,
        k_val_train=1,
        k_val_val=15,
        k_val_test=15,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=2000,
        meta_learning_rate=0.001,
        report_validation_frequency=250,
        log_train_images_after_iteration=1000,
        number_of_tasks_val=100,
        number_of_tasks_test=1000,
        clip_gradients=True,
        experiment_name='celeba_noise_std_1.2',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    gan_sampling.train(iterations=20000)
    gan_sampling.evaluate(iterations=50, use_val_batch_statistics=True, seed=42)


if __name__ == '__main__':
    run_celeba()
