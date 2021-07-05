"""
    Defines:
        - Abstract AbstractClassifier class. Individual classifiers should be subclasses thereof.

    Author: Dr Dmitry A. Duev
    February 2020

    Deep Neural Networks

    Author: Dr Dmitry A. Duev
    February/March 2020
"""

import os
import json
import datetime

from abc import ABC, abstractmethod
from collections import defaultdict

import kerastuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from ast import literal_eval

from tqdm import tqdm
from sklearn.model_selection import train_test_split


class AbstractClassifier(ABC):

    def __init__(self, name):
        # classifier name: label_classifier
        self.name = name
        # model that will be trained and evaluated
        self.model = None
        # metadata needed to set up the classifier
        self.meta = defaultdict(str)

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

class DNN(AbstractClassifier):
    """
        Baseline model with a statically-defined graph
    """

    def setup(self, features_shape=(41, ), dmdt_shape=(26, 26, 1), dense_branch=True, conv_branch=True,
              loss='binary_crossentropy', optimizer='adam', callbacks=('early_stopping', 'tensorboard'),
              tag=None, logdir='logs', histogram_freq=0, **kwargs):

        tf.keras.backend.clear_session()

        self.model = self.build_model(features_shape=features_shape, dmdt_shape=dmdt_shape,
                                      dense_branch=dense_branch, conv_branch=conv_branch)

        self.meta['loss'] = loss
        if optimizer == 'adam':
            lr = kwargs.get('lr', 3e-4)
            beta_1 = kwargs.get('beta_1', 0.9)
            beta_2 = kwargs.get('beta_2', 0.999)
            epsilon = kwargs.get('epsilon', 1e-7)  # None?
            decay = kwargs.get('decay', 0.0)
            amsgrad = kwargs.get('amsgrad', 3e-4)
            self.meta['optimizer'] = tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                                                              epsilon=epsilon, decay=decay, amsgrad=amsgrad)
        elif optimizer == 'sgd':
            lr = kwargs.get('lr', 3e-4)
            momentum = kwargs.get('momentum', 0.9)
            decay = kwargs.get('epsilon', 1e-6)
            nesterov = kwargs.get('nesterov', True)
            self.meta['optimizer'] = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        else:
            print('Could not recognize optimizer, using Adam with default params')
            self.meta['optimizer'] = tf.keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,
                                                              epsilon=1e-7, decay=0.0, amsgrad=False)
        # self.meta['epochs'] = epochs
        # self.meta['patience'] = patience
        # self.meta['weight_per_class'] = weight_per_class

        self.meta['metrics'] = [tf.keras.metrics.TruePositives(name='tp'),
                                tf.keras.metrics.FalsePositives(name='fp'),
                                tf.keras.metrics.TrueNegatives(name='tn'),
                                tf.keras.metrics.FalseNegatives(name='fn'),
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.AUC(name='auc'),
                                ]

        self.meta['callbacks'] = []
        # self.meta['callbacks'] = [TqdmCallback(verbose=1)]
        for callback in set(callbacks):
            if callback == 'early_stopping':
                # halt training if no gain in <validation loss> over <patience> epochs
                monitor = kwargs.get('monitor', 'val_loss')
                patience = kwargs.get('patience', 10)
                restore_best_weights = kwargs.get('restore_best_weights', True)
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                                           patience=patience,
                                                                           restore_best_weights=restore_best_weights)
                self.meta['callbacks'].append(early_stopping_callback)

            elif callback == 'tensorboard':
                # logs for TensorBoard:
                if tag:
                    log_tag = f'{self.name.replace(" ", "_")}-{tag}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                else:
                    log_tag = f'{self.name.replace(" ", "_")}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                logdir_tag = os.path.join('logs', log_tag)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(logdir_tag, log_tag),
                                                                      histogram_freq=histogram_freq)
                self.meta['callbacks'].append(tensorboard_callback)

        self.model.compile(optimizer=self.meta['optimizer'],
                           loss=self.meta['loss'],
                           metrics=self.meta['metrics'])

    @staticmethod
    def build_model(features_shape: tuple = (41, ), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_2')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            x_conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.Flatten()(x_conv)

            x_conv = tf.keras.layers.Dense(256, activation='relu', name='conv_fc_1')(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
            x_conv = tf.keras.layers.Dense(32, activation='relu', name='conv_fc_2')(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m

    def train(self, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val,
              epochs=300, class_weight=None, verbose=0):

        if not class_weight:
            # all our problems here are binary classification ones:
            class_weight = {i: 1 for i in range(2)}

        self.meta['history'] = self.model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch_train,
                                              validation_data=val_dataset, validation_steps=steps_per_epoch_val,
                                              class_weight=class_weight, callbacks=self.meta['callbacks'],
                                              verbose=verbose)

    def evaluate(self, test_dataset, **kwargs):
        return self.model.evaluate(test_dataset, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def load(self, path_model, **kwargs):
        self.model = tf.keras.models.load_model(path_model, **kwargs)

    def save(self, output_path='./', output_format='hdf5', tag=None):

        assert output_format in ('SavedModel', 'hdf5'), 'unknown output format'

        output_name = self.name if not tag else f'{self.name}.{tag}'

        if (output_path != './') and (not os.path.exists(output_path)):
            os.makedirs(output_path)

        if output_format == 'SavedModel':
            self.model.save(os.path.join(output_path, output_name))
        elif output_format == 'hdf5':
            self.model.save(os.path.join(output_path, f'{output_name}.h5'))


class DNN_v2(DNN):

    @staticmethod
    def build_model(features_shape: tuple = (41,), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_2')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            x_conv = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m


class DNN_v3(DNN):

    @staticmethod
    def build_model(features_shape: tuple = (41, ), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            # x_dense = tf.keras.layers.Dense(128, activation='relu', name='dense_fc_2')(x_dense)
            # x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            # x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_3')(x_dense)
            x_dense = tf.keras.layers.Dense(64, activation='relu', name='dense_fc_3')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            # batch norm momentum
            batch_norm_momentum = 0.2

            # kernel size
            ks = (3, 3)

            f1, f2, f3 = 16, 32, 64

            x_conv = tf.keras.layers.SeparableConv2D(f1, ks, name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv1')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(f1, ks, name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv2')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            # save to add a skip connection later
            x_shortcut = x_conv

            x_conv = tf.keras.layers.SeparableConv2D(f2, ks, name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv3')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(f2, ks, name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv4')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='conv_conv_5')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv5')(x_conv)

            # shortcut path
            x_shortcut = tf.keras.layers.SeparableConv2D(
                f3, kernel_size=(1, 1), strides=(2, 2), padding='valid', name='conv_conv_shortcut',
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            )(x_shortcut)
            x_shortcut = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, name='bn_conv_shortcut')(x_shortcut)

            x_shortcut = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_shortcut)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation
            x_conv = tf.keras.layers.add([x_shortcut, x_conv])
            x_conv = tf.keras.layers.Activation('relu')(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m


class DNN_v4(DNN):

    @staticmethod
    def build_model(features_shape: tuple = (41, ), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            # x_dense = tf.keras.layers.Dense(128, activation='relu', name='dense_fc_2')(x_dense)
            # x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_3')(x_dense)
            # x_dense = tf.keras.layers.Dense(64, activation='relu', name='dense_fc_3')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            # batch norm momentum
            batch_norm_momentum = 0.2

            # kernel size
            ks = (3, 3)

            f1, f2, f3 = 32, 32, 32

            x_conv = tf.keras.layers.SeparableConv2D(f1, ks, name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv1')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(f1, ks, name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv2')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            # save to add a skip connection later
            x_shortcut = x_conv

            x_conv = tf.keras.layers.SeparableConv2D(f2, ks, name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv3')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(f2, ks, name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv4')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='conv_conv_5')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv5')(x_conv)

            # shortcut path
            x_shortcut = tf.keras.layers.SeparableConv2D(
                f3, kernel_size=(1, 1), strides=(2, 2), padding='valid', name='conv_conv_shortcut',
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            )(x_shortcut)
            x_shortcut = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, name='bn_conv_shortcut')(x_shortcut)

            x_shortcut = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_shortcut)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation
            x_conv = tf.keras.layers.add([x_shortcut, x_conv])
            x_conv = tf.keras.layers.Activation('relu')(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m


class DNN_v5(DNN):

    @staticmethod
    def build_model(features_shape: tuple = (41, ), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(512, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(128, activation='relu', name='dense_fc_2')(x_dense)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_3')(x_dense)
            # x_dense = tf.keras.layers.Dense(64, activation='relu', name='dense_fc_3')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            # batch norm momentum
            batch_norm_momentum = 0.2

            # kernel size
            ks = (3, 3)

            f1, f2, f3 = 32, 32, 32

            x_conv = tf.keras.layers.SeparableConv2D(f1, ks, name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv1')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(f1, ks, name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv2')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            # save to add a skip connection later
            x_shortcut = x_conv

            x_conv = tf.keras.layers.SeparableConv2D(f2, ks, name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv3')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(f2, ks, name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv4')(x_conv)
            x_conv = tf.keras.layers.Activation('relu')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='conv_conv_5')(x_conv)
            x_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv5')(x_conv)

            # shortcut path
            x_shortcut = tf.keras.layers.SeparableConv2D(
                f3, kernel_size=(1, 1), strides=(2, 2), padding='valid', name='conv_conv_shortcut',
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            )(x_shortcut)
            x_shortcut = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, name='bn_conv_shortcut')(x_shortcut)

            x_shortcut = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_shortcut)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation
            x_conv = tf.keras.layers.add([x_shortcut, x_conv])
            x_conv = tf.keras.layers.Activation('relu')(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m


class DNN_v6(DNN):
    # 20200623 hp search run

    @staticmethod
    def build_model(features_shape: tuple = (41,), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(288, activation='relu', name='dense_fc_2')(x_dense)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(64, activation='relu', name='dense_fc_3')(x_dense)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_4')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            x_conv = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(32, activation='relu', name='fc_1')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m


class DNN_v7(DNN):
    # 20200623 hp search run

    @staticmethod
    def build_model(features_shape: tuple = (40,), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True, **kwargs):

        if (not dense_branch) and (not conv_branch):
            raise ValueError('model must have at least one branch')

        features_input = tf.keras.Input(shape=features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=dmdt_shape, name='dmdt')

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(224, activation='relu', name='dense_fc_1')(features_input)
            x_dense = tf.keras.layers.Dropout(0.4)(x_dense)
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_2')(x_dense)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_3')(x_dense)
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_4')(x_dense)

        # CNN branch to digest dmdt
        if conv_branch:
            x_conv = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv_conv_1')(dmdt_input)
            x_conv = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv_conv_2')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.15)(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', name='conv_conv_3')(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', name='conv_conv_4')(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # head
        x = tf.keras.layers.Dense(32, activation='relu', name='fc_1')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='fc_2')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='fc_3')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='fc_4')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m


class TunableModel(kt.HyperModel):

    def __init__(
            self, name=None, tunable=True,
            features_shape=(41,), dmdt_shape=(26, 26, 1),
            dense_branch=True, conv_branch=True,
            meta=None
    ):
        super(TunableModel, self).__init__(name=name, tunable=tunable)

        if True not in (dense_branch, conv_branch):
            raise ValueError('Mmodel must have at least one branch, dense or conv')

        self.features_shape = features_shape
        self.dmdt_shape = dmdt_shape
        self.dense_branch = dense_branch
        self.conv_branch = conv_branch

        self.meta = meta

    def build(self, hp):

        features_input = tf.keras.Input(shape=self.features_shape, name='features')
        dmdt_input = tf.keras.Input(shape=self.dmdt_shape, name='dmdt')

        # dense branch to digest features
        if self.dense_branch:
            for i in range(hp.Int('dense_layers', 1, 4, default=1)):
                x_dense = tf.keras.layers.Dense(
                    hp.Int(f'dense_fc_{i+1}', min_value=32, max_value=512, step=32, default=256),
                    activation='relu',
                )(features_input)
                x_dense = tf.keras.layers.Dropout(
                    hp.Float(f'dense_dropout_{i+1}', min_value=0.15, max_value=0.55, step=0.1, default=0.25)
                )(x_dense)
            x_dense = tf.keras.layers.Dense(
                hp.Int(f'dense_fc_last', min_value=32, max_value=512, step=32, default=256),
                activation='relu'
            )(x_dense)

        # CNN branch to digest dmdt
        if self.conv_branch:
            for i in range(hp.Int('conv_blocks', 1, 5, default=2)):
                filters = hp.Int('conv_filters_' + str(i+1), 16, 64, step=16)
                x_conv = tf.keras.layers.SeparableConv2D(filters, (3, 3), activation='relu')(dmdt_input)
                x_conv = tf.keras.layers.SeparableConv2D(filters, (3, 3), activation='relu')(x_conv)
                x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
                x_conv = tf.keras.layers.Dropout(
                    hp.Float(f'conv_dropout_{i + 1}', min_value=0.15, max_value=0.55, step=0.1, default=0.25)
                )(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if self.dense_branch and self.conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif self.dense_branch:
            x = x_dense
        elif self.conv_branch:
            x = x_conv

        # more dense layers? yeschyo odnu polosu dobavim i vsyo poyedet
        for i in range(hp.Int('dense_layers', 0, 2, default=1)):
            x = tf.keras.layers.Dense(hp.Int(f'fc_{i+1}', min_value=16, max_value=64, step=16, default=32),
                                      activation='relu')(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        m.compile(optimizer=self.meta['optimizer'], loss=self.meta['loss'], metrics=self.meta['metrics'])

        return m


class DNNTunable(DNN):
    """
    Tunable model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hypermodel = None
        self.tuner = None

    def setup(
        self, features_shape=(41,), dmdt_shape=(26, 26, 1), dense_branch=True, conv_branch=True,
        loss='binary_crossentropy', optimizer='adam', patience=2, callbacks=('early_stopping', 'tensorboard'),
        tag=None, logdir='logs', histogram_freq=0, **kwargs
    ):

        tf.keras.backend.clear_session()

        self.meta['loss'] = loss
        if optimizer == 'adam':
            lr = kwargs.get('lr', 3e-4)
            beta_1 = kwargs.get('beta_1', 0.9)
            beta_2 = kwargs.get('beta_2', 0.999)
            epsilon = kwargs.get('epsilon', 1e-7)  # None?
            decay = kwargs.get('decay', 0.0)
            amsgrad = kwargs.get('amsgrad', 3e-4)
            self.meta['optimizer'] = tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                                                              epsilon=epsilon, decay=decay, amsgrad=amsgrad)
        elif optimizer == 'sgd':
            lr = kwargs.get('lr', 3e-4)
            momentum = kwargs.get('momentum', 0.9)
            decay = kwargs.get('epsilon', 1e-6)
            nesterov = kwargs.get('nesterov', True)
            self.meta['optimizer'] = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        else:
            print('Could not recognize optimizer, using Adam with default params')
            self.meta['optimizer'] = tf.keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,
                                                              epsilon=1e-7, decay=0.0, amsgrad=False)
        # self.meta['epochs'] = epochs
        # self.meta['patience'] = patience
        # self.meta['weight_per_class'] = weight_per_class

        self.meta['metrics'] = [tf.keras.metrics.TruePositives(name='tp'),
                                tf.keras.metrics.FalsePositives(name='fp'),
                                tf.keras.metrics.TrueNegatives(name='tn'),
                                tf.keras.metrics.FalseNegatives(name='fn'),
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.AUC(name='auc'),
                                ]

        self.meta['callbacks'] = []
        # self.meta['callbacks'] = [TqdmCallback(verbose=1)]
        for callback in set(callbacks):
            if callback == 'early_stopping':
                # halt training if no gain in <validation loss> over <patience> epochs
                monitor = kwargs.get('monitor', 'val_loss')
                patience = kwargs.get('patience', 10)
                restore_best_weights = kwargs.get('restore_best_weights', True)
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                                           patience=patience,
                                                                           restore_best_weights=restore_best_weights)
                self.meta['callbacks'].append(early_stopping_callback)

            elif callback == 'tensorboard':
                # logs for TensorBoard:
                if tag:
                    log_tag = f'{self.name.replace(" ", "_")}-{tag}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                else:
                    log_tag = f'{self.name.replace(" ", "_")}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                logdir_tag = os.path.join('logs', log_tag)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(logdir_tag, log_tag),
                                                                      histogram_freq=histogram_freq)
                self.meta['callbacks'].append(tensorboard_callback)

        self.hypermodel = self.build_model(
            features_shape=features_shape, dmdt_shape=dmdt_shape,
            dense_branch=dense_branch, conv_branch=conv_branch,
        )

        objective = kwargs.get('tuner_objective', 'val_loss')
        max_trials = kwargs.get('tuner_max_trials', 10)

        self.tuner = kt.RandomSearch(
            self.hypermodel,
            objective=objective,
            max_trials=max_trials,
        )

    def build_model(self, features_shape: tuple = (41,), dmdt_shape: tuple = (26, 26, 1),
                    dense_branch: bool = True, conv_branch: bool = True):

        m = TunableModel(
            name=self.name, tunable=True,
            features_shape=features_shape, dmdt_shape=dmdt_shape,
            dense_branch=dense_branch, conv_branch=conv_branch,
            meta=self.meta
        )

        return m

    def train(self, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val,
              epochs=300, class_weight=None, verbose=False):

        if not class_weight:
            # all our problems here are binary classification ones:
            class_weight = {i: 1 for i in range(2)}

        self.meta['history'] = self.tuner.search(
            train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch_train,
            validation_data=val_dataset, validation_steps=steps_per_epoch_val,
            class_weight=class_weight, callbacks=self.meta['callbacks'],
            verbose=verbose
        )

        # retrieve the best model
        self.model = self.tuner.get_best_models(num_models=1)[0]
        # print(self.tuner.results_summary())

class Dataset(object):

    def __init__(
        self, path_dataset,
        path_labels: str = '../labels',
        features=(
            'ad', 'chi2red', 'f1_a', 'f1_amp', 'f1_b',
            'f1_BIC', 'f1_phi0', 'f1_power', 'f1_relamp1', 'f1_relamp2',
            'f1_relamp3', 'f1_relamp4', 'f1_relphi1', 'f1_relphi2', 'f1_relphi3',
            'f1_relphi4', 'i60r', 'i70r', 'i80r', 'i90r', 'inv_vonneumannratio', 'iqr',
            'median', 'median_abs_dev',
            # 'n',
            'norm_excess_var', 'norm_peak_to_peak_amp', 'pdot', 'period', 'roms', 'significance',
            'skew', 'smallkurt', 'stetson_j', 'stetson_k', 'sw', 'welch_i', 'wmean',
            'wstd', 'n_ztf_alerts', 'mean_ztf_alert_braai'
        ),
        verbose=False,
        **kwargs
    ):  
        """
        load csv file produced by labels*.ipynb

        :param tag:
        :param path_labels:
        :param features:
        :param verbose:
        """
        self.verbose = verbose
        self.features = features

        if self.verbose:
            print(f'Loading {path_dataset}...')
        self.df_ds = pd.read_csv(path_dataset)
        if self.verbose:
            print(self.df_ds[list(features)].describe())

        dmdt = []
        if self.verbose:
            print('Moving dmdt\'s to a dedicated numpy array...')
            for i in tqdm(self.df_ds.itertuples(), total=len(self.df_ds)):
                #if i.Index > 10: continue
                var = np.asarray(literal_eval(self.df_ds['dmdt'][i.Index]))
                if not var.shape == (26,26):
                    var = np.zeros((26,26))
                dmdt.append(var)
        else:
            for i in self.df_ds.itertuples():
                var = np.asarray(literal_eval(self.df_ds['dmdt'][i.Index]))
                if not var.shape == (26,26):
                    var = np.zeros((26,26))
                dmdt.append(var)
        self.dmdt = np.dstack(dmdt)
        self.dmdt = np.transpose(self.dmdt, (2, 0, 1))
        #self.dmdt = np.expand_dims(self.dmdt, axis=-1)

        # drop in df_ds:
        self.df_ds.drop(columns='dmdt', inplace=True)
        self.df_ds.fillna(0, inplace=True)

    @staticmethod
    def threshold(a, t: float = 0.5):
        b = np.zeros_like(a)
        b[np.array(a) > t] = 1
        return b

    def make(
        self, target_label: str = 'variable', threshold: float = 0.5, balance=None, weight_per_class: bool = True,
        test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
        path_norms=None, batch_size: int = 256, shuffle_buffer_size: int = 256, epochs: int = 300,
        **kwargs
    ):
        """
        make datasets for target_label

        :param target_label:
        :param threshold:
        :param balance:
        :param weight_per_class:
        :param test_size:
        :param val_size:
        :param random_state:
        :param path_norms: json file with norms to use to normalize features. if None, norms are computed
        :param batch_size
        :param shuffle_buffer_size
        :param epochs
        :return:
        """

        # Note: Dataset.from_tensor_slices method requires the target variable to be of the int type.
        # TODO: see what to do about it when trying label smoothing in the future.
        if isinstance(target_label, list):
            for tar in target_label[1:]:
                wc2 = self.df_ds[tar] >= 0.7
                self.df_ds.loc[wc2, target_label[0]] = 1
            target_label = target_label[0]

        # target = np.asarray(list(map(int, np.rint(self.df_ds[target_label].values))))
        target = np.asarray(list(map(int, self.threshold(self.df_ds[target_label].values, t=threshold))))

        self.target = np.expand_dims(target, axis=1)

        neg, pos = np.bincount(target.flatten())
        total = neg + pos
        if self.verbose:
            print(f'Examples:\n  Total: {total}\n  Positive: {pos} ({100 * pos / total:.2f}% of total)\n')

        w_pos = np.rint(self.df_ds[target_label].values) == 1
        index_pos = self.df_ds.loc[w_pos].index
        if target_label == 'variable':
            # 'variable' is a special case: there is an explicit 'non-variable' label:
            w_neg = np.asarray(list(map(int, self.threshold(self.df_ds['non-variable'].values, t=threshold)))) == 1
        else:
            w_neg = ~w_pos
        index_neg = self.df_ds.loc[w_neg].index

        # balance positive and negative examples if there are more negative than positive?
        index_neg_dropped = None
        if balance:
            neg_sample_size = int(np.sum(w_pos) * balance)
            index_neg = self.df_ds.loc[w_neg].sample(n=neg_sample_size, random_state=1).index
            index_neg_dropped = self.df_ds.loc[list(set(self.df_ds.loc[w_neg].index) - set(index_neg))].index

        ds_indexes = index_pos.to_list() + index_neg.to_list()

        # Train/validation/test split (we will use an 81% / 9% / 10% data split by default):

        train_indexes, test_indexes = train_test_split(ds_indexes, shuffle=True,
                                                       test_size=test_size, random_state=random_state)
        train_indexes, val_indexes = train_test_split(train_indexes, shuffle=True,
                                                      test_size=val_size, random_state=random_state)

        # Normalize features (dmdt's are already L2-normalized) (?using only the training samples?).
        # Obviously, the same norms will have to be applied at the testing and serving stages.

        # load/compute feature norms:
        if not path_norms or not os.path.isfile(path_norms):
            norms = {feature: np.linalg.norm(self.df_ds.loc[ds_indexes, feature]) for feature in self.features}
            for feature, norm in norms.items():
                if np.isnan(norm) or norm == 0.0:
                    norms[feature] = 1.0
            if self.verbose:
                print('Computed feature norms:\n', norms)
            with open(path_norms, 'w') as f:
                json.dump(norms, f)
        else:
            with open(path_norms, 'r') as f:
                norms = json.load(f)
            if self.verbose:
                print(f'Loaded feature norms from {path_norms}:\n', norms)

        for feature, norm in norms.items():
            self.df_ds[feature] /= norm

        # replace zeros with median values
        if kwargs.get('zero_to_median', False):
            for feature in norms.keys():
                if feature in ('pdot', 'n_ztf_alerts'):
                    continue
                wz = self.df_ds[feature] == 0.0
                if wz.sum() > 0:
                    if feature == 'mean_ztf_alert_braai':
                        median = 0.5
                    else:
                        median = self.df_ds.loc[~wz, feature].median()
                    self.df_ds.loc[wz, feature] = median

        # make tf.data.Dataset's:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[train_indexes, self.features].values, 'dmdt': self.dmdt[train_indexes]},
             target[train_indexes])
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[val_indexes, self.features].values, 'dmdt': self.dmdt[val_indexes]},
             target[val_indexes])
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[test_indexes, self.features].values, 'dmdt': self.dmdt[test_indexes]},
             target[test_indexes])
        )
        dropped_negatives = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[index_neg_dropped, self.features].values,
              'dmdt': self.dmdt[index_neg_dropped]},
             target[index_neg_dropped])
        ) if balance else None

        # Shuffle and batch the datasets:
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epochs)
        val_dataset = val_dataset.batch(batch_size).repeat(epochs)
        test_dataset = test_dataset.batch(batch_size)

        dropped_negatives = dropped_negatives.batch(batch_size) if balance else None

        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'dropped_negatives': dropped_negatives,
        }

        indexes = {
            'train': np.array(train_indexes),
            'val': np.array(val_indexes),
            'test': np.array(test_indexes),
            'dropped_negatives': np.array(index_neg_dropped.to_list()) if index_neg_dropped is not None else None,
        }

        # How many steps per epoch?

        steps_per_epoch_train = len(train_indexes) // batch_size - 1
        steps_per_epoch_val = len(val_indexes) // batch_size - 1
        steps_per_epoch_test = len(test_indexes) // batch_size - 1

        steps_per_epoch = {'train': steps_per_epoch_train,
                           'val': steps_per_epoch_val,
                           'test': steps_per_epoch_test}
        if self.verbose:
            print(f'Steps per epoch: {steps_per_epoch}')

        # Weight training data depending on the number of samples?
        # Very useful for imbalanced classification, especially when in the cases with a small number of examples.

        if weight_per_class:
            # weight data class depending on number of examples?
            # num_training_examples_per_class = np.array([len(target) - np.sum(target), np.sum(target)])
            num_training_examples_per_class = np.array([len(index_neg), len(index_pos)])

            assert 0 not in num_training_examples_per_class, 'found class without any examples!'

            # fewer examples -- larger weight
            weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
            normalized_weight = weights / np.max(weights)

            class_weight = {i: w for i, w in enumerate(normalized_weight)}

        else:
            # working with binary classifiers only
            class_weight = {i: 1 for i in range(2)}

        return datasets, indexes, steps_per_epoch, class_weight

