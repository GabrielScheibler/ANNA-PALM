import tensorflow  as tf

class ReversibleNet():

    def __init__(self, num_blocks, num_channels):
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.weights_per_layer = 6
        self.weight_list = self.def_rev_block_weights(self.num_channels)
        self.weight_ta = tf.TensorArray(dtype=tf.float32, size=len(self.weight_list), dynamic_size=False, clear_after_read=False,
                            infer_shape=False)
        for i in range(len(self.weight_list)):
            self.weight_ta = self.weight_ta.write(i, self.weight_list[i])


    def def_rev_block_weights(self, channels):
        """
        Defines weights for the revnet.
        Args:
            channels: channel number of the input tensors
        Returns:
            weight_list: list of all weights for the revnet, sorted alphabetically by variable-scope/name
        """
        weight_list = []
        for i in range(self.num_blocks):
            with tf.variable_scope("rev_core_%03d" % (i + 1)):
                weight_list = rev_block_weights(channels, weight_list)
        return weight_list


    def forward_pass_activations(self, inputs):
        """
        Generates forward pass of the revnet, using more memory, but able to be backpropagated through
        Args:
            inputs: tuple of two same sized input tensors
        Returns:
            layer_rev_out: tuple of the two ouput tensors of the last network layer
        """
        print("revnet")
        print(inputs[0].shape)
        print(inputs[1].shape)

        for i in range(self.num_blocks):
            layer_weights = []
            for j in range(self.weights_per_layer):
                layer_weights.append(self.weight_ta.read(i * self.weights_per_layer + j))

            in_1, in_2 = inputs
            out_1, out_2 = rev_block(in_1, in_2, layer_weights, reverse=False)
            out_1.set_shape(in_1.get_shape())
            out_2.set_shape(in_1.get_shape())
            inputs = (out_1, out_2)

        return inputs


    def backward_pass_activations(self, inputs):
        """
        Generates forward pass of the revnet, using more memory, but able to be backpropagated through
        Args:
            inputs: tuple of two same sized input tensors
        Returns:
            layer_rev_out: tuple of the two ouput tensors of the last network layer
        """
        print("revnet")
        print(inputs[0].shape)
        print(inputs[1].shape)

        for i in range(self.num_blocks - 1, -1, -1):
            layer_weights = []
            for j in range(self.weights_per_layer):
                layer_weights.append(self.weight_ta.read(i * self.weights_per_layer + j))

            in_1, in_2 = inputs
            out_1, out_2 = rev_block(in_1, in_2, layer_weights, reverse=True)
            out_1.set_shape(in_1.get_shape())
            out_2.set_shape(in_1.get_shape())
            inputs = (out_1, out_2)

        return inputs


    def forward_pass(self, inputs):
        """
        Generates forward pass of the revnet
        Args:
            inputs: tuple of two same sized input tensors
        Returns:
            layer_rev_out: tuple of the two ouput tensors of the last network layer
        """
        print("revnet")
        print(inputs[0].shape)
        print(inputs[1].shape)

        def loop_body(layer_index, inputs, weights):
            layer_weights = []
            for i in range(self.weights_per_layer):
                layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

            in_1, in_2 = inputs
            out_1, out_2 = rev_block(in_1, in_2, layer_weights, reverse=False)
            out_1.set_shape(in_1.get_shape())
            out_2.set_shape(in_1.get_shape())
            output = (out_1, out_2)

            return [layer_index + 1, output, weights]

        _, layer_rev_out, ta = tf.while_loop(lambda i, _, __: i < self.num_blocks, loop_body,
                                             loop_vars=[tf.constant(0), inputs, self.weight_ta], parallel_iterations=1,
                                             back_prop=False)
        return layer_rev_out


    def backward_pass(self, inputs):
        """
        Generates backward pass of the revnet
        Args:
            inputs: tuple of two same sized input tensors going into the last layer
        Returns:
            layer_rev_out: tuple of the two ouput tensors of the first network layer
        """
        print("revnet_backward")
        print(inputs[0].shape)
        print(inputs[1].shape)

        def loop_body(layer_index, inputs, weights):
            layer_weights = []
            for i in range(self.weights_per_layer):
                layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

            in_1, in_2 = inputs
            out_1, out_2 = rev_block(in_1, in_2, layer_weights, reverse=True)
            out_1.set_shape(in_1.get_shape())
            out_2.set_shape(in_1.get_shape())
            output = (out_1, out_2)

            return [layer_index - 1, output, weights]

        _, layer_rev_out, ta = tf.while_loop(lambda i, _, __: i >= 0, loop_body,
                                             loop_vars=[tf.constant(self.num_blocks - 1), inputs, self.weight_ta], parallel_iterations=1,
                                             back_prop=False)
        return layer_rev_out


    def compute_revnet_gradients_of_forward_pass(self, y1, y2, dy1, dy2):
        """
        Computes gradients.
        Args:
          y1: Output activation 1.
          y2: Output activation 2.
          dy1: Output gradient 1.
          dy2: Output gradient 2.
        Returns:
          dx1: Input gradient 1.
          dx2: Input gradient 2.
          grads_and_vars: List of tuples of gradient and variable.
        """
        with tf.name_scope("manual_gradients"):
            print("Manually building gradient graph.")
            #tf.get_variable_scope().reuse_variables()

            grads_list = []

            weights = self.weight_ta
            weights_grads = tf.TensorArray(dtype=tf.float32, size=len(self.weight_list), dynamic_size=False,
                                           clear_after_read=False, infer_shape=False)

            outputs = (y1, y2)
            output_grads = (dy1, dy2)

            def loop_body(layer_index, outputs, output_grads, weights, weights_grads):
                layer_weights = []
                for i in range(self.weights_per_layer):
                    layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

                (inputs, input_grads, layer_weights_grads) = self.backprop_layer_forward_pass(outputs, output_grads,
                                                                                              layer_weights)

                for i in range(self.weights_per_layer):
                    weights_grads = weights_grads.write(layer_index * self.weights_per_layer + i,
                                                        tf.squeeze(layer_weights_grads[i]))

                inputs[1].set_shape(outputs[1].get_shape())
                inputs[0].set_shape(outputs[0].get_shape())
                input_grads[1].set_shape(output_grads[1].get_shape())
                input_grads[0].set_shape(output_grads[0].get_shape())
                return [layer_index - 1, inputs, input_grads, weights, weights_grads]

            _, inputs, input_grads, _, weights_grads = tf.while_loop(lambda i, *_: i >= 0, loop_body,
                                                                     [tf.constant(self.num_blocks - 1), outputs,
                                                                      output_grads, weights, weights_grads],
                                                                     parallel_iterations=1, back_prop=False)

            for i in range(self.num_blocks * self.weights_per_layer):
                grads_list.append(weights_grads.read(index=i))

            return inputs, input_grads, list(zip(grads_list, self.weight_list))


    def backprop_layer_forward_pass(self, outputs, output_grads, layer_weights):
        """
        Computes gradient for one layer.
        Args:
          outputs: Outputs of the layer
          output_grads: gradients for the layer outputs
          layer_weights: all weights for the rev_layer
        Returns:
          inputs: inputs of the layer
          input_grads: gradients for the layer inputs
          weight_grads: gradients for the layer weights
        """
        # First , reverse the layer to r e t r i e v e inputs
        y1, y2 = outputs[0], outputs[1]
        # F_weights , G_weights = tf.split(layer_weights, num_or_size_splits=2, axis=1)
        split_index = int(len(layer_weights) / 2)
        f_weights = layer_weights[0:split_index]
        g_weights = layer_weights[split_index:]

        y1_grad = output_grads[0]
        y2_grad = output_grads[1]


        """
        # backprop implementation 1
        x1, x2 = rev_block(y1, y2, layer_weights, reverse=True)
        x1, x2 = tf.stop_gradient(x1), tf.stop_gradient(x2)
        y1, y2 = rev_block(x1, x2, layer_weights, reverse=False)

        dd1 = tf.gradients(y2, [y1] + g_weights, y2_grad, gate_gradients=True)
        dy2_y1 = dd1[0]
        dy1_plus = dy2_y1 + y1_grad
        dgw = dd1[1:]
        dd2 = tf.gradients(y1, [x1, x2] + f_weights, dy1_plus, gate_gradients=True)
        dx1 = dd2[0]
        dx2 = dd2[1]
        dfw = dd2[2:]
        dx2 += tf.gradients(x2, x2, y2_grad, gate_gradients=True)[0]

        x1_grad = dx1
        x2_grad = dx2
        
        dw_list = list(dfw) + list(dgw)
        """

        """
        # backprop implementation 2
        x1, x2 = rev_block(y1, y2, layer_weights, reverse=True)
        x1, x2 = tf.stop_gradient(x1), tf.stop_gradient(x2)

        y1, y2 = rev_block(x1, x2, layer_weights, reverse=False)

        grads = tf.gradients([y1, y2], [x1, x2] + f_weights + g_weights, [y1_grad, y2_grad], gate_gradients=False)
        x1_grad, x2_grad, dw_list = grads[0], grads[1], grads[2:]
        """

        # backprop implementation 3
        with tf.variable_scope("rev_block"):
            z1_stop = tf.stop_gradient(y1)
            with tf.variable_scope("g"):
                G_z1 = rev_nn(z1_stop, g_weights)
                x2 = y2 - G_z1
                x2_stop = tf.stop_gradient(x2)
            with tf.variable_scope("f"):
                F_x2 = rev_nn(x2_stop, f_weights)
                x1 = y1 - F_x2
                x1_stop = tf.stop_gradient(x1)

        z1 = x1_stop + F_x2
        y2 = x2_stop + G_z1
        y1 = z1

        z1_g_grad = tf.gradients(y2, [z1_stop] + g_weights, y2_grad)
        z1_grad = z1_g_grad[0] + y1_grad
        g_grads = z1_g_grad[1:]
        x2_f_grad = tf.gradients(y1, [x2_stop] + f_weights, z1_grad)
        x2_grad = x2_f_grad[0] + y2_grad
        f_grads = x2_f_grad[1:]
        x1_grad = z1_grad

        dw_list = list(f_grads) + list(g_grads)

        inputs = (x1, x2)
        input_grads = (x1_grad, x2_grad)
        weight_grads = dw_list
        return inputs, input_grads, weight_grads


    def compute_revnet_gradients_of_backward_pass(self, x1, x2, dx1, dx2):
        """
        Computes gradients.
        Args:
          y1: Output activation 1.
          y2: Output activation 2.
          dy1: Output gradient 1.
          dy2: Output gradient 2.
        Returns:
          dx1: Input gradient 1.
          dx2: Input gradient 2.
          grads_and_vars: List of tuples of gradient and variable.
        """
        with tf.name_scope("manual_gradients"):
            print("Manually building gradient graph.")
            tf.get_variable_scope().reuse_variables()

            grads_list = []

            weights = self.weight_ta
            weights_grads = tf.TensorArray(dtype=tf.float32, size=len(self.weight_list), dynamic_size=False,
                                           clear_after_read=False, infer_shape=False)

            outputs = (x1, x2)
            output_grads = (dx1, dx2)

            def loop_body(layer_index, outputs, output_grads, weights, weights_grads):
                layer_weights = []
                for i in range(self.weights_per_layer):
                    layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

                (inputs, input_grads, layer_weights_grads) = self.backprop_layer_backward_pass(outputs, output_grads, layer_weights)

                for i in range(self.weights_per_layer):
                    weights_grads = weights_grads.write(layer_index * self.weights_per_layer + i, tf.squeeze(layer_weights_grads[i]))

                inputs[1].set_shape(outputs[1].get_shape())
                inputs[0].set_shape(outputs[0].get_shape())
                input_grads[1].set_shape(output_grads[1].get_shape())
                input_grads[0].set_shape(output_grads[0].get_shape())
                return [layer_index + 1, inputs, input_grads, weights, weights_grads]

            _, inputs, input_grads, _, weights_grads = tf.while_loop(lambda i, *_: i <= self.num_blocks - 1, loop_body,
                                                                     [tf.constant(0), outputs,
                                                                      output_grads, weights, weights_grads],
                                                                     parallel_iterations=1, back_prop=False)

            for i in range(self.num_blocks * self.weights_per_layer):
                grads_list.append(weights_grads.read(index=i))

            return inputs, input_grads, list(zip(grads_list, self.weight_list))


    def backprop_layer_backward_pass(self, outputs, output_grads, layer_weights):
        """
        Computes gradient for one layer.
        Args:
          outputs: Outputs of the layer
          output_grads: gradients for the layer outputs
          layer_weights: all weights for the rev_layer
        Returns:
          inputs: inputs of the layer
          input_grads: gradients for the layer inputs
          weight_grads: gradients for the layer weights
        """
        # First , reverse the layer to r e t r i e v e inputs
        x1, x2 = outputs[0], outputs[1]
        split_index = int(len(layer_weights) / 2)
        f_weights = layer_weights[0:split_index]
        g_weights = layer_weights[split_index:]

        x1_grad, x2_grad = output_grads

        """
        # backprop implementation 3 - not functioning
        with tf.variable_scope("rev_block"):
            z2_stop = tf.stop_gradient(x2)
            with tf.variable_scope("f"):
                F_z2 = rev_nn(z2_stop, f_weights)
                y1 = x1 + F_z2
                y1_stop = tf.stop_gradient(y1)
            with tf.variable_scope("g"):
                G_y1 = rev_nn(z2_stop, g_weights)
                y2 = x2 + G_y1
                y2_stop = tf.stop_gradient(y2)

        z2 = y2_stop - G_y1
        x1 = y1_stop - F_z2
        x2 = z2

        z2_f_grad = tf.gradients(x1, [z2_stop] + f_weights, x1_grad)
        z2_grad = z2_f_grad[0] + x2_grad
        f_grads = z2_f_grad[1:]
        y1_g_grad = tf.gradients(y2, [y1_stop] + g_weights, z2_grad)
        print(y1_g_grad)
        y1_grad = y1_g_grad[0] + x1_grad
        g_grads = y1_g_grad[1:]
        y2_grad = z2_grad

        dw_list = list(f_grads) + list(g_grads)
        """

        # backprop implementation 2
        y1, y2 = rev_block(x1, x2, layer_weights, reverse=False)
        y1, y2 = tf.stop_gradient(y1), tf.stop_gradient(y2)

        x1, x2 = rev_block(y1, y2, layer_weights, reverse=True)

        grads = tf.gradients([x1, x2], [y1, y2] + f_weights + g_weights, [x1_grad, x2_grad], gate_gradients=False)
        y1_grad, y2_grad, dw_list = grads[0], grads[1], grads[2:]


        inputs = (y1, y2)
        input_grads = (y1_grad, y2_grad)
        weight_grads = dw_list
        return inputs, input_grads, weight_grads


def rev_block(in_1, in_2, weights, reverse):
    split_index = int(len(weights) / 2)
    weights_f = weights[0:split_index]
    weights_g = weights[split_index:]
    with tf.variable_scope("rev_block"):
        if reverse:
            # x2 = y2 - NN2(y1)
            with tf.variable_scope("g"):
                out_2 = in_2 - rev_nn(in_1, weights_g)

            # x1 = y1 - NN1(x2)
            with tf.variable_scope("f"):
                out_1 = in_1 - rev_nn(out_2, weights_f)
        else:
            # y1 = x1 + NN1(x2)
            with tf.variable_scope("f"):
                out_1 = in_1 + rev_nn(in_2, weights_f)

            # y2 = x2 + NN2(y1)
            with tf.variable_scope("g"):
                out_2 = in_2 + rev_nn(out_1, weights_g)

        return [out_1, out_2]


def rev_nn(in_1, weights):
    out = in_1
    out = rev_batchnorm(out, weights[0:2])
    out = rev_lrelu(out, 0.2)
    out = rev_conv3x3(out, tf.squeeze(weights[2]))
    out.set_shape(in_1.get_shape())
    return out


def rev_conv3x3(batch_input, weight):
    with tf.variable_scope("conv3x3"):
        filter = weight
        padded_in_1 = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        out_1 = tf.nn.conv2d(padded_in_1, filter, [1, 1, 1, 1], padding="VALID")
        return out_1


def rev_lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def rev_batchnorm(input, weights):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        offset = weights[0]
        scale = weights[1]
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2, 3], keep_dims=True)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                               variance_epsilon=variance_epsilon)
        return normalized


def rev_block_weights(channels, weights):
    with tf.variable_scope("rev_block"):
        with tf.variable_scope("f"):
            weights = rev_nn_weights(channels, weights)
        with tf.variable_scope("g"):
            weights = rev_nn_weights(channels, weights)
        return weights


def rev_nn_weights(channels, weights):
    weights = rev_batchnorm_weights(channels, weights)
    weights = rev_conv3x3_weights(channels, weights)
    return weights


def rev_conv3x3_weights(channels, weights):
    with tf.variable_scope("conv3x3"):
        weights.append(tf.get_variable("filter", [3, 3, channels, channels], dtype=tf.float32, initializer=tf.zeros_initializer()))
        return weights


def rev_batchnorm_weights(channels, weights):
    with tf.variable_scope("batchnorm"):
        weights.append(tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer()))
        weights.append(tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02)))
        return weights
