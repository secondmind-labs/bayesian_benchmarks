import gpflow
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from scipy.stats import norm


class RegressionModel:
    def __init__(self, is_test=False, seed=0):
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 10000
                small_iterations = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self._model = None
        self._model_objective = None

    def fit(self, X, Y):
        num_data, input_dim = X.shape

        if num_data > self.ARGS.num_inducing:
            Z, _ = kmeans2(X, self.ARGS.num_inducing, minit='points')
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.ARGS.num_inducing - num_data, input_dim)],
                               axis=0)

        # make model if necessary
        if self._model is None:
            data = (tf.Variable(X, trainable=False), tf.Variable(Y, trainable=False))
            kernel = gpflow.kernels.SquaredExponential(lengthscale=float(input_dim)**0.5)
            # Gaussian likelihood: use SGPR
            self._model = gpflow.models.SGPR(data, kernel, inducing_variable=Z,
                                             noise_variance=self.ARGS.initial_likelihood_var)

            @tf.function(autograph=False)
            def objective():
                return - self._model.log_marginal_likelihood()
            self._model_objective = objective

        # we might have new data
        self._model.data[0].assign(X)
        self._model.data[1].assign(Y)
        self._model.inducing_variable.Z.assign(Z)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(self._model_objective, self._model.trainable_variables,
                     options=dict(maxiter=self.ARGS.iterations))

    def predict(self, Xs):
        return self._model.predict_y(Xs)

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)


class ClassificationModel:
    def __init__(self, K, is_test=False, seed=0):
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 5000
                small_iterations = 1000
                initial_likelihood_var = 0.01

        self.ARGS = ARGS
        self.K = K
        self._model = None
        self._model_objective = None

    def fit(self, X, Y):
        num_data, input_dim = X.shape

        if num_data > self.ARGS.num_inducing:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]
        else:
            Z = X.copy()

        if self._model is None:
            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent = self.K

            kernel = gpflow.kernels.SquaredExponential(lengthscale=float(input_dim) ** 0.5)
            self._model = gpflow.models.SVGP(kernel, lik,
                                             inducing_variable=Z,
                                             whiten=False,
                                             num_latent=num_latent)

            @tf.function(autograph=False)
            def objective(data):
                return - self._model.log_marginal_likelihood(data)
            self._model_objective = objective

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self._model.inducing_variable.Z.assign(Z)

        num_outputs = self._model.q_sqrt.shape[0]
        self._model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self._model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        data = (tf.constant(X), tf.constant(Y))

        def objective_closure():
            return self._model_objective(data)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, self._model.trainable_variables,
                     options=dict(maxiter=iters))

    def predict(self, Xs):
        m, v = self._model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to one-hot
            return np.concatenate([1 - m, m], axis=1)
        else:
            return m
