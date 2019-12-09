import gpflow
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

try:
    from tqdm import trange
except ImportError:
    trange = range
    
class RegressionModel:
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
        self._model = None
        self._model_objective = None
        self._adam_opt = None
        self._natgrad_opt = None

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
            kernel = gpflow.kernels.SquaredExponential(lengthscale=float(input_dim)**0.5)
            lik = gpflow.likelihoods.Gaussian(variance=self.ARGS.initial_likelihood_var)
            self._model = gpflow.models.SVGP(kernel, likelihood=lik, inducing_variable=Z)

            @tf.function(autograph=False)
            def objective(data):
                return - self._model.log_marginal_likelihood(data)
            self._model_objective = objective

            self._model.q_mu.trainable = False
            self._model.q_sqrt.trainable = False
            self._natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=self.ARGS.gamma)
            self._adam_opt = tf.optimizers.Adam(learning_rate=self.ARGS.adam_lr)

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self._model.inducing_variable.Z.assign(Z)
        num_outputs = self._model.q_sqrt.shape[0]
        self._model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self._model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        batch_size = np.minimum(self.ARGS.minibatch_size, num_data)
        data = (X, Y)
        data_minibatch = tf.data.Dataset.from_tensor_slices(data) \
            .prefetch(num_data).repeat().shuffle(num_data) \
            .batch(batch_size)
        data_minibatch_it = iter(data_minibatch)

        def objective_closure() -> tf.Tensor:
            batch = next(data_minibatch_it)
            return self._model_objective(batch)

        variational_params = [(self._model.q_mu, self._model.q_sqrt)]

        @tf.function
        def natgrad_step():
            self._natgrad_opt.minimize(objective_closure, var_list=variational_params)

        @tf.function
        def adam_step():
            self._adam_opt.minimize(objective_closure, var_list=self._model.trainable_variables)

        for _ in trange(iters):
            natgrad_step()
            adam_step()

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
        self._model = None
        self._model_objective = None
        self._opt = None

    def fit(self, X, Y):
        num_data, input_dim = X.shape

        if num_data > self.ARGS.num_inducing:
            Z, _ = kmeans2(X, self.ARGS.num_inducing, minit='points')
        else:
            Z = X.copy()

        if self._model is None:
            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent = self.K

            kernel = gpflow.kernels.SquaredExponential(lengthscale=float(input_dim)**0.5)
            self._model = gpflow.models.SVGP(kernel, likelihood=lik, inducing_variable=Z,
                                             whiten=False, num_latent=num_latent)

            @tf.function(autograph=False)
            def objective(data):
                return - self._model.log_marginal_likelihood(data)
            self._model_objective = objective

            self._opt = tf.optimizers.Adam(self.ARGS.adam_lr)

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self._model.inducing_variable.Z.assign(Z)
        num_outputs = self._model.q_sqrt.shape[0]
        self._model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self._model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        batch_size = np.minimum(self.ARGS.minibatch_size, num_data)
        data = (X, Y)
        data_minibatch = tf.data.Dataset.from_tensor_slices(data) \
            .prefetch(num_data).repeat().shuffle(num_data) \
            .batch(batch_size)
        data_minibatch_it = iter(data_minibatch)

        def objective_closure() -> tf.Tensor:
            batch = next(data_minibatch_it)
            return self._model_objective(batch)

        @tf.function
        def adam_step():
            self._opt.minimize(objective_closure, var_list=self._model.trainable_variables)

        for _ in trange(iters):
            adam_step()

    def predict(self, Xs):
        m, v = self._model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], axis=1)
        else:
            return m
