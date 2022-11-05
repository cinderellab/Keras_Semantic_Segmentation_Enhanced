import os
from pprint import pprint
import datetime
import numpy as np
import sys
sys.path.append('.')
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils.vis_utils import plot_model

from core.configures import training_config, net_config, augment_config
from core.nets import SemanticSegmentationModel
from core.utils.data_utils.data_generator import ImageDataGenerator


def parse_training_args():
    def learning_rate_schedule(epoch):
        lr_base = training_config.base_lr
        lr_min = training_config.min_lr
        epochs = training_config.epoch
        lr_power = training_config.lr_power
        lr_cycle = training_config.lr_cycle
        mode = training_config.lr_mode
        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
        elif mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)

        elif mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1
        elif mode is 'cosine_cycle':
            lr = ((lr_base - lr_min) / 2) * (np.cos(2 * np.pi * (epoch % lr_cycle / lr_cycle)) + 1)
        elif mode is 'none':
            lr = lr_base
        else:
            raise ValueError("Invalid learning rate schedule mode: {}. Expected 'power_decay', 'exp_decay', 'adam', "
                             "'progressive_drops', 'cosine_cycle'.".format(mode))

        return lr

    losses = {
                'binary_crossentropy': 'binary_crossentropy',
                'categorical_crossentropy': 'categorical_crossentropy'
              }
    metrics = {
                    'acc': 'acc'
               }

    training_config.loss = losses[training_config.loss_name]
    training_config.metric = metrics[training_config.metric_name]

    if training_config.optimizer_name.lower() == "adam":
        training_config.optimizer = Adam(training_config.base_lr)
    elif training_config.optimizer_name.lower() == "rmsprop":
        training_config.optimizer = RMSprop(training_config.base_lr)
    else:
        training_config.optimizer = SGD(training_config.base_lr, momentum=0.9)

    if not os.path.exists("{}/models".format(training_config.workspace)):
        os.mkdir("{}/models".format(training_config.workspace))
    if not os.path.exists("{}/logs".format(training_config.workspace)):
        os.mkdir("{}/logs".format(training_config.workspace))
    training_config.load_model_name = "{}/models/{}.h5".format(
        training_config.workspace, training_config.old_model_version)
    training_config.save_model_name = "{}/models/{}.h5".format(
        training_config.workspace, training_config.new_model_version)

    training_config.callbacks = list()
    training_config.callba