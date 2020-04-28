import tensorflow as tf
from tensorflow import layers

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class AE(object):

    def __init__(self, n_out, is_training, n_filters=64, name='U-Net_2D', trainable=True):
        """
        Class for Auto-Encoder architecture. This is a 2D version, which means it only employs
        bi-dimensional convolution and strides. This implementation also uses batch normalization after each conv layer.
        :param n_out: (int) number of channels for the network output. For instance, to predict a binary mask you must
                        use n_out=2 (one-hot encoding); to predict a grayscale image you must use n_out=1
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout, batch normalization, ecc. (which behave differently
                        at train and test time)
        :param n_filters: (int) number of filters in the first layer. Default=64 (as in the vanilla UNet)
        :param name: (string) name scope for the AE
        :param trainable: whether this model is used for training or restoring

        - - - - - - - - - - - - - - - -
        Notice that:
          - this implementation works for incoming tensors with shape [None, N, M, K], where N and M must be divisible
            by 16 without any rest (in fact, there are 4 pooling layers with kernels 2x2 --> input reduced to:
            [None, N/16, M/16, K'])
          - the output of the network does not have activation linear
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire AE model:
            output = AE(n_out, is_training).build(incoming_tensor)

            # build the AE with access to the internal code:
            ae = AE(n_out, is_training)
            encoder = AE.build_encoder(incoming_tensor)
            code = AE.build_bottleneck(encoder_outputs)
            decoder = AE.build_decoder(code)
            output = AE.build_output(decoder)

        """

        self.incoming = None
        self.n_out = n_out
        self.is_training = is_training
        self.nf = n_filters
        self.name = name
        self.code = None
        self.trainable = trainable

    def build(self, incoming, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        """

        # with tf.variable_scope(self.name, reuse=reuse):
        encoder = self.build_encoder(incoming)
        code = self.build_bottleneck(encoder)
        decoder = self.build_decoder(code)
        output = self.build_output(decoder)

        # import sys
        print("!!!! Debug Here:")
        print("encoder.shape: ", encoder[3].shape)
        print("code.shape: ", code.shape)
        print("decoder.shape: ", decoder.shape)
        print("output.shape: ", output.shape)
        # sys.exit()

        return output

    def build_encoder(self, incoming, reuse=tf.AUTO_REUSE):
        """ Encoder layers """
        # check for compatible input dimensions
        shape = incoming.get_shape().as_list()
        assert not shape[1] % 16
        assert not shape[2] % 16
        self.incoming = incoming

        with tf.variable_scope(self.name + '/Encoder', reuse=reuse):
            en_brick_0 = self._encode_brick(self.incoming, self.nf, self.is_training,
                                            scope='encode_brick_0', trainable=self.trainable)
            en_brick_1 = self._encode_brick(en_brick_0, 2 * self.nf, self.is_training,
                                            scope='encode_brick_1', trainable=self.trainable)
            en_brick_2 = self._encode_brick(en_brick_1, 4 * self.nf, self.is_training,
                                            scope='encode_brick_2', trainable=self.trainable)
            en_brick_3 = self._encode_brick(en_brick_2, 8 * self.nf, self.is_training,
                                            scope='encode_brick_3', trainable=self.trainable)

        return en_brick_0, en_brick_1, en_brick_2, en_brick_3

    def build_bottleneck(self, encoder, reuse=tf.AUTO_REUSE):
        """ Central layers """
        en_brick_0, en_brick_1, en_brick_2, en_brick_3 = encoder

        with tf.variable_scope(self.name + '/Bottleneck', reuse=reuse):
            code = self._bottleneck_brick(en_brick_3, 16 * self.nf, self.is_training,
                                          scope='code', trainable=self.trainable)
            self.code = code

        return code

    def build_decoder(self, code, reuse=tf.AUTO_REUSE):
        """ Decoder layers """

        with tf.variable_scope(self.name + '/Decoder', reuse=reuse):
            dec_brick_0 = self._decode_brick(code, 8 * self.nf, self.is_training,
                                             scope='decode_brick_0', trainable=self.trainable)
            dec_brick_1 = self._decode_brick(dec_brick_0, 4 * self.nf, self.is_training,
                                             scope='decode_brick_1', trainable=self.trainable)
            dec_brick_2 = self._decode_brick(dec_brick_1, 2 * self.nf, self.is_training,
                                             scope='decode_brick_2', trainable=self.trainable)
            dec_brick_3 = self._decode_brick(dec_brick_2, self.nf, self.is_training,
                                             scope='decode_brick_3', trainable=self.trainable)

        return dec_brick_3

    def build_output(self, decoder, reuse=tf.AUTO_REUSE):
        """ Output layers """
        # output linear

        with tf.variable_scope(self.name + '/Output', reuse=reuse):
            output = self._output_layer(decoder, n_channels_out=self.n_out, scope='output', trainable=self.trainable)
        return output

    @staticmethod
    def _encode_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Encoding brick: conv --> conv --> max pool.
        """
        with tf.variable_scope(scope):
            conv1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv1_bn = layers.batch_normalization(conv1, training=is_training, trainable=trainable)
            conv1_act = tf.nn.relu(conv1_bn)

            conv2 = layers.conv2d(conv1_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv2_bn = layers.batch_normalization(conv2, training=is_training, trainable=trainable)
            conv2_act = tf.nn.relu(conv2_bn)

            pool = layers.max_pooling2d(conv2_act, pool_size=2, strides=2, padding='same')

        return pool

    @staticmethod
    def _decode_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Decoding brick: deconv (up-pool) --> conv --> conv.
        """
        with tf.variable_scope(scope):
            _, old_height, old_width, __ = incoming.get_shape()
            new_height, new_width = 2.0 * old_height, 2.0 * old_width
            upsampled = tf.image.resize_nearest_neighbor(incoming, size=[new_height, new_width])
            conv1t = layers.conv2d(upsampled, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                   kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv1t_bn = layers.batch_normalization(conv1t, training=is_training, trainable=trainable)
            conv1t_act = tf.nn.relu(conv1t_bn)

            conv2 = layers.conv2d(conv1t_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv2_bn = layers.batch_normalization(conv2, training=is_training, trainable=trainable)
            conv2_act = tf.nn.relu(conv2_bn)

            conv3 = layers.conv2d(conv2_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv3_bn = layers.batch_normalization(conv3, training=is_training, trainable=trainable)
            conv3_act = tf.nn.relu(conv3_bn)

        return conv3_act

    @staticmethod
    def _bottleneck_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Code brick: conv --> conv .
        """
        with tf.variable_scope(scope):
            code1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            code1_bn = layers.batch_normalization(code1, training=is_training, trainable=trainable)
            code1_act = tf.nn.relu(code1_bn)

            code2 = layers.conv2d(code1_act, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            code2_bn = layers.batch_normalization(code2, training=is_training, trainable=trainable)
            code2_act = tf.nn.relu(code2_bn)

        return code2_act

    @staticmethod
    def _output_layer(incoming, n_channels_out, scope, trainable=True):
        """ Output layer: conv .
        """
        with tf.variable_scope(scope):
            output = layers.conv2d(incoming, filters=n_channels_out, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            # activation = linear
        return output
