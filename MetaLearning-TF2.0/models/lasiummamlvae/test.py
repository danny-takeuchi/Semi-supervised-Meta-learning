import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

ds = tfds.load('omniglot', split='train', shuffle_files=True)
