import numpy as np
import tensorflow as tf



class Policy:
    def __init__(self):
        self.W = self.__rand_initialization()

        self.__tensorflow_init()

    def choose_action(self, s, greedy=True):
        """ Choose action based on a probability vector p. If greedy -> argmax, if not -> sample

        :param s: state
        :type s: ndarray
        :param greedy: boolean signifying greedy or not
        :type greedy: bool
        :return: action
        :rtype: int
        """
        s = np.reshape(s, (1, 4))
        p = self.__softmax(np.dot(s, self.W))
        if greedy:
            return np.argmax(p)
        else:
            return np.random.choice([0, 1], 1, p=p)[0]

    def __tensorflow_init(self):
        """ Building of a TensorFlow computational graph

        """
        self.state_input = tf.placeholder(tf.float64, shape=[None, 4])
        self.action_input = tf.placeholder(tf.float64, shape=[None, 2])

        self.W_tf = tf.Variable(self.W, [4, 2])
        self.output = tf.matmul(self.state_input, self.W_tf)
        self.output_sm = tf.nn.softmax(self.output)
        self.output_sm_for_action = tf.reduce_sum(tf.multiply(self.output_sm, self.action_input))
        self.objective = tf.log(self.output_sm_for_action)

        self.grad_tf = tf.gradients(self.objective, [self.W_tf])[0]

    def __softmax(self, x):
        """ Computes a normalized softmax (so that np.exp() does not overflow)

        :param x: vector to be transformed
        :type x: ndarray
        """


        x = np.reshape(x, (2,))
        a = max(x)
        return np.exp(x - a) / np.sum(np.exp(x - a))

    def __rand_initialization(self):
        """
        Initializer of the Neural Net Weights

        """
        return np.random.normal(10, 1, size=(4, 2))

    def __compute_log_policy_gradient(self, s, a, analytical=True):
        """
        Computes the gradient (log pi), both explicit formula and tensorflow derivations are available

        :param s: state
        :type s: ndarray
        :param a: action
        :type a: int
        :param analytical: if True, analytical formula is used (MUCH FASTER). if False,  TensorFlow automatic gradient
        :type analytical: bool
        :return: gradient
        :rtype: ndarray
        """
        if analytical:
            grad = np.zeros((4, 2))
            product = np.dot(s, self.W)
            c = np.sum(np.exp(product))

            if a == 0:  # origina a == 0
                grad[:, 0] = s * (1 - np.exp(product[0]) / c)
                grad[:, 1] = -s * np.exp(product[1]) / c
            else:
                grad[:, 0] = -s * np.exp(product[0]) / c
                grad[:, 1] = s * (1 - np.exp(product[1]) / c)

            return grad

        else:
            # TensorFlow
            # convert action to one hot representation
            a_ohr = np.zeros((1, 2))
            a_ohr[0][a] = 1

            s = np.reshape(s, (1, 4))

            # DEFINE OPS
            init = tf.global_variables_initializer()
            assign_op = self.W_tf.assign(self.W)

            # Start a TF session
            sess = tf.Session()

            # Run Opds
            sess.run(init)
            sess.run(assign_op)

            grad_tf = sess.run(self.grad_tf, feed_dict={self.state_input: s,
                                                        self.action_input: a_ohr})
            sess.close()

            return grad_tf

    def update(self, s, a, return_estimate, step_size):
        """
        Update of the neural network weights W based on the gradient and other parameters

        :param s: state
        :type s: ndarray
        :param a: action
        :type a: int
        :param return_estimate: estimate of expected return
        :type return_estimate: int
        :param step_size: step size
        :type step_size: float
        """
        log_pol_grad = self.__compute_log_policy_gradient(s, a)
        self.W += step_size * log_pol_grad * return_estimate
