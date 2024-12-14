from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import tensorflow as tf

import models.utils as utils
from models import model_registry
from models.distributions_con import CategoricalPd
from models.tf_v1_model import TFV1Model
from models.utils import conv_to_fc, conv, cirbasicblock, fc

__all__ = ['ACModel', 'ACMLPModel', 'ACCNNModel']


class ACModel(TFV1Model, ABC):
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        with tf.variable_scope(model_id):
            self.x_ph = utils.placeholder(shape=observation_space, dtype=tf.float32)
            self.encoded_x_ph = tf.to_float(self.x_ph)
            self.a_ph = utils.placeholder(dtype=tf.float32, shape=(self.action_space))

        self.mu = None
        self.sigma = None
        self.vf = None

        super(ACModel, self).__init__(observation_space, action_space, config, model_id, scope=model_id,
                                      *args, **kwargs)

        pd = CategoricalPd(self.mu, self.sigma)
        self.action = pd.sample()
        self.neglogp_a = pd.neglogp(self.action)
        self.neglogp_a_ph = pd.neglogp(self.a_ph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def forward(self, states: Any, *args, **kwargs) -> Any:
        return self.sess.run([self.mu, self.action, self.vf, self.neglogp_a], feed_dict={self.x_ph: states})

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass


@model_registry.register('acmlp_con')
class ACMLPModel(ACModel):

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('pi'):
                self.mu = utils.mlp(self.encoded_x_ph, [64, 64, self.action_space], tf.tanh)
                # self.sigma = utils.mlp(self.encoded_x_ph, [64, 64, self.action_space], tf.tanh)

            with tf.variable_scope('v'):
                self.vf = tf.squeeze(utils.mlp(self.encoded_x_ph, [64, 64, 1], tf.tanh), axis=1)


@model_registry.register('accnn_con')
class ACCNNModel(ACModel, ABC):

    def build(self, *args, **kwargs) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('cnn_base'):
                # scaled_images = tf.cast(self.encoded_x_ph, tf.float32) / 255.
                input_images = tf.cast(self.encoded_x_ph, tf.float32)
                activ = tf.nn.relu
                
                # outstem = activ(conv(input_images, 'c1', nf=16, rf=8, stride=4, pad='SAME', init_scale=np.sqrt(2)))
                # outstem = tf.nn.max_pool(outstem,[1,2,2,1],[1,1,1,1], padding='VALID')
                # outresnet = cirbasicblock(outstem,"rb1_1",16,1)
                # outresnet = cirbasicblock(outresnet,"rb1_2",16,1)
                # outresnet = cirbasicblock(outresnet,"rb2_1",16,2)
                # outresnet = cirbasicblock(outresnet,"rb2_2",16,1)
                # outresnet = cirbasicblock(outresnet,"rb3_1",32,2)
                # outresnet = cirbasicblock(outresnet,"rb3_2",32,1)
                # outresnet = cirbasicblock(outresnet,"rb4_1",32,2)
                # outresnet = cirbasicblock(outresnet,"rb4_2",32,1)
                # outresnet = conv_to_fc(outresnet)

                outstem = activ(conv(input_images, 'c1', nf=16, rf=8, stride=4, pad='SAME', init_scale=np.sqrt(2)))
                outstem = tf.nn.max_pool(outstem,[1,2,2,1],[1,1,1,1], padding='VALID')
                outstem = activ(conv(outstem, 'c2', nf=16, rf=8, stride=4, pad='SAME', init_scale=np.sqrt(2)))
                outstem = conv_to_fc(outstem)

                latent = activ(fc(outstem, 'fc1', nh=64, init_scale=np.sqrt(2)))

            with tf.variable_scope('pi'):
                pih1 = activ(fc(latent, 'pi_fc1', 64, init_scale=1))
                pih2 = activ(fc(pih1, 'pi_fc2', 64, init_scale=1))
                self.mu = tf.tanh(fc(pih2, 'pi_fc3', self.action_space, init_scale=1))
                # self.logits = logits_without_mask + 1000. *tf.to_float(self.legal_action-1)

            with tf.variable_scope('v'):
                vh1 = activ(fc(latent, 'v_fc1', 64, init_scale=1))
                vh2 = activ(fc(vh1, 'v_fc2', 64, init_scale=1))
                self.vf = tf.squeeze(fc(vh2, 'v_fc3', 1, init_scale=1), axis=1)
