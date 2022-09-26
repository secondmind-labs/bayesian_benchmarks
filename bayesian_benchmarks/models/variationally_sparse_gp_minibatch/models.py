import gpflow
import tensorflow as tf
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
        self.model = None
        self.model_objective = None
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
        if self.model is None:
            lengthscales = np.full(input_dim, float(input_dim)**0.5)
            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
            lik = gpflow.likelihoods.Gaussian(variance=self.ARGS.initial_likelihood_var)
            self.model = gpflow.models.SVGP(kernel, likelihood=lik, inducing_variable=Z, num_data=num_data)

            gpflow.set_trainable(self.model.q_mu, False)
            gpflow.set_trainable(self.model.q_sqrt, False)
            self._natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=self.ARGS.gamma)
            self._adam_opt = tf.optimizers.Adam(learning_rate=self.ARGS.adam_lr)

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.inducing_variable.Z.assign(Z)
        num_outputs = self.model.q_sqrt.shape[0]
        self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        if num_data < self.ARGS.minibatch_size:
            model_objective = self.model.training_loss_closure((X, Y))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)) \
                .prefetch(num_data).repeat().shuffle(num_data)
            train_iter = iter(train_dataset.batch(self.ARGS.minibatch_size))
            model_objective = self.model.training_loss_closure(train_iter)

        variational_params = [(self.model.q_mu, self.model.q_sqrt)]

        @tf.function
        def natgrad_step():
            self._natgrad_opt.minimize(model_objective, var_list=variational_params)

        @tf.function
        def adam_step():
            self._adam_opt.minimize(model_objective, var_list=self.model.trainable_variables)

        for _ in trange(iters):
            natgrad_step()
            adam_step()

    def predict(self, Xs):
        return self.model.predict_y(Xs)

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
        self.model = None
        self.model_objective = None
        self.opt = None

    def fit(self, X, Y):
        num_data, input_dim = X.shape

        if num_data > self.ARGS.num_inducing:
            Z, _ = kmeans2(X, self.ARGS.num_inducing, minit='points')
        else:
            Z = X.copy()

        if self.model is None:
            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent_gps = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent_gps = self.K

            lengthscales = np.full(input_dim, float(input_dim)**0.5)
            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
            self.model = gpflow.models.SVGP(kernel, likelihood=lik, inducing_variable=Z,
                                            num_latent_gps=num_latent_gps)

            self.opt = tf.optimizers.Adam(self.ARGS.adam_lr)

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.inducing_variable.Z.assign(Z)
        num_outputs = self.model.q_sqrt.shape[0]
        self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        if num_data < self.ARGS.minibatch_size:
            model_objective = self.model.training_loss_closure((X, Y))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)) \
                .prefetch(num_data).repeat().shuffle(num_data)
            train_iter = iter(train_dataset.batch(self.ARGS.minibatch_size))
            model_objective = self.model.training_loss_closure(train_iter)

        @tf.function
        def adam_step():
            self.opt.minimize(objective_closure, var_list=self.model.trainable_variables)

        for _ in trange(iters):
            adam_step()

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], axis=1)
        else:
            return m
