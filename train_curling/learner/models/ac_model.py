from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import tensorflow as tf

import models.utils as utils
from models import model_registry
from models.distributions import CategoricalPd
from models.tf_v1_model import TFV1Model
from models.utils import conv, fc, conv_to_fc, placeholder

__all__ = ['ACModel', 'ACMLPModel', 'ACCNNModel']


class ACModel(TFV1Model, ABC):
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        with tf.variable_scope(model_id):
            self.x_ph = placeholder(shape=observation_space, dtype=tf.float32)
            self.encoded_x_ph = tf.to_float(self.x_ph)
            self.a_ph = placeholder(shape=2, dtype=tf.int32)

        self.logits_force = None
        self.logits_angle = None
        self.vf = None

        super(ACModel, self).__init__(observation_space, action_space, config, model_id, scope=model_id,
                                      *args, **kwargs)

        pd_force = CategoricalPd(self.logits_force)
        pd_angle = CategoricalPd(self.logits_angle)
        
        action_force = pd_force.sample()
        action_angle = pd_angle.sample()
        
        neglogp_force = pd_force.neglogp(action_force)
        neglogp_angle = pd_angle.neglogp(action_angle)
        self.neglogp = neglogp_force + neglogp_angle
        
        action_force = tf.reshape(action_force, [-1, 1])
        action_angle = tf.reshape(action_angle, [-1, 1])
        self.action = tf.concat([action_force, action_angle], axis=1)

        force_ph, angle_ph = tf.split(self.a_ph, [1,1], axis=1)
        force_ph = tf.reshape(force_ph, [-1])
        angle_ph = tf.reshape(angle_ph, [-1])
        neglogp_force_ph = pd_force.neglogp(force_ph)
        neglogp_angle_ph = pd_angle.neglogp(angle_ph)
        self.neglogp_a = neglogp_angle_ph + neglogp_force_ph

        entropy_force = pd_force.entropy()
        entropy_angle = pd_angle.entropy()
        self.entropy = entropy_angle + entropy_force

        self.sess.run(tf.global_variables_initializer())

    def forward(self, states: Any, *args, **kwargs) -> Any:
        return self.sess.run([self.action, self.vf, self.neglogp], feed_dict={self.x_ph: states})

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass

@model_registry.register('acmlp')
class ACMLPModel(ACModel):

    def build(self) -> None:
        activ = tf.nn.relu
        with tf.variable_scope(self.scope):
            me, team, enemy, center, first_to_move, throws_left = tf.split(self.encoded_x_ph, [2,6,8,2,1,1] , axis = 1)

            team_cen = tf.concat([team, center], axis=1)
            with tf.variable_scope('team_center_process'):
                h1 = activ(fc(team_cen, 'fc1', 8, init_scale=1))
                h2 = activ(fc(h1, 'fc2', 8, init_scale=1))
                out1 = activ(fc(h2,'fc3',6, init_scale=1))

            enemy_cen = tf.concat([enemy, center], axis=1)
            with tf.variable_scope('enemy_center_process'):
                h1 = activ(fc(enemy_cen, 'fc1', 10, init_scale=1))
                h2 = activ(fc(h1, 'fc2', 10, init_scale=1))
                out2 = activ(fc(h2,'fc3',8, init_scale=1))

            out = tf.concat([me, out1, out2, first_to_move, throws_left], axis=1)

            with tf.variable_scope('pi_force'):
                pih1 = activ(fc(out, 'pi_fc1', 18, init_scale=1))
                pih2 = activ(fc(pih1, 'pi_fc2', 18, init_scale=1))
                logits_force = fc(pih2, 'pi_fc3', self.action_space, init_scale=1)
                self.logits_force = logits_force

            with tf.variable_scope('pi_angle'):
                pih1 = activ(fc(out, 'pi_fc1', 18, init_scale=1))
                pih2 = activ(fc(pih1, 'pi_fc2', 18, init_scale=1))
                logits_angle = fc(pih2, 'pi_fc3', self.action_space, init_scale=1)
                self.logits_angle = logits_angle

            with tf.variable_scope('v'):
                h1 = activ(fc(out, 'fc1', 16, init_scale=1))
                h2 = activ(fc(h1, 'fc2', 8, init_scale=1))
                self.vf = tf.squeeze(fc(h2,'fc3', 1, init_scale=1), axis=1)



@model_registry.register('accnn')
class ACCNNModel(ACModel, ABC):

    def build(self, *args, **kwargs) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('cnn_base'):
                scaled_images = tf.cast(self.encoded_x_ph, tf.float32) / 255.
                activ = tf.nn.relu
                h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
                h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
                h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
                h3 = conv_to_fc(h3)
                latent = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
                latent = tf.layers.flatten(latent)

            with tf.variable_scope('pi'):
                self.logits = fc(latent, 'pi', self.action_space, init_scale=0.01)

            with tf.variable_scope('v'):
                self.vf = tf.squeeze(fc(latent, 'vf', 1), axis=1)
