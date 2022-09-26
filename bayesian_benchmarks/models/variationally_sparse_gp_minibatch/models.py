import gpflow
import tensorflow as tf
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

class RegressionModel(object):
    def __init__(self, is_test=False, seed=0):
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                adam_lr = 0.01
                gamma = 0.1
                minibatch_size = 100
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 10000
                small_iterations = 1000
                adam_lr = 0.01
                gamma = 0.1
                minibatch_size = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        N, D = X.shape
        _, K = Y.shape

        if N > self.ARGS.num_inducing:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.ARGS.num_inducing - N, D)], 0)

        # make model if necessary
        if not self.model:
            lengthscales = np.full(D, float(D)**0.5)
            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
            lik = gpflow.likelihoods.Gaussian(variance=self.ARGS.initial_likelihood_var)
            self.model = gpflow.models.SVGP(kernel, lik, inducing_variable=Z, num_data=N)

            self.variational_params = [(self.model.q_mu, self.model.q_sqrt)]
            gpflow.set_trainable(self.model.q_mu, False)
            gpflow.set_trainable(self.model.q_sqrt, False)
            self.natgrad = gpflow.optimizers.NaturalGradient(gamma=self.ARGS.gamma)
            self.adam = tf.optimizers.Adam(self.ARGS.adam_lr)

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.inducing_variable.Z.assign(Z)

        self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, K)))
        self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [K, 1, 1]))

        if N > self.ARGS.minibatch_size:
            loss = self.model.training_loss_closure((X, Y))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)
            train_iter = iter(train_dataset.batch(self.ARGS.minibatch_size))
            loss = self.model.training_loss_closure(train_iter)

        @tf.function
        def natgrad_step():
            self.natgrad.minimize(loss, var_list=self.variational_params)

        @tf.function
        def adam_step():
            self.adam.minimize(loss, var_list=self.model.trainable_variables)

        for _ in range(iters):
            natgrad_step()
            adam_step()

    def predict(self, Xs):
        return self.model.predict_y(Xs)

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)


class ClassificationModel(object):
    def __init__(self, K, is_test=False, seed=0):
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                adam_lr = 0.01
                minibatch_size = 100
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 10000
                small_iterations = 1000
                adam_lr = 0.01
                minibatch_size = 1000
        self.ARGS = ARGS

        self.K = K
        self.model = None

    def fit(self, X, Y):
        N, D = X.shape
        Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0] if N > self.ARGS.num_inducing else X.copy()

        if not self.model:
            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent = self.K

            lengthscales = np.full(D, float(D)**0.5)
            kernel = gpflow.kernels.RBF(lengthscales=lengthscales)
            self.model = gpflow.models.SVGP(kernel, lik,
                                            feat=Z,
                                            num_latent_gps=num_latent,
                                            num_data=N)

            self.opt = tf.optimizers.Adam(self.ARGS.adam_lr)

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.inducing_variable.Z.assign(Z)

        num_outputs = self.model.q_sqrt.shape[0]
        self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        if N < self.ARGS.minibatch_size:
            loss = self.model.training_loss_closure((X, Y))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)
            train_iter = iter(train_dataset.batch(self.ARGS.minibatch_size))
            loss = self.model.training_loss_closure(train_iter)

        @tf.function
        def step():
            self.opt.minimize(loss, self.model.trainable_variables)

        for _ in range(iters):
            step()

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m
