import gpflow
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
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 10000
                small_iterations = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        N, D = X.shape
        if N > self.ARGS.num_inducing:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.ARGS.num_inducing - N, D)], 0)

        # make model if necessary
        if not self.model:
            lengthscales = np.full(D, float(D)**0.5)
            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)

            self.model = gpflow.models.SGPR((X, Y), kernel, inducing_variable=Z, noise_variance=self.ARGS.initial_likelihood_var)
            self.opt = gpflow.optimizers.Scipy()

        # we might have new data
        self.model.data = (X, Y)
        self.model.inducing_variable.Z.assign(Z)

        self.opt.minimize(self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=self.ARGS.iterations))

    def predict(self, Xs):
        return self.model.predict_y(Xs)

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, L = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, L) * (v ** 0.5)


class ClassificationModel(object):
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
            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)

            self.model = gpflow.models.SVGP(kernel, lik,
                                            inducing_variable=Z,
                                            num_latent_gps=num_latent,
                                            num_data=N)

            self.opt = gpflow.optimizers.Scipy()

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.inducing_variable.Z.assign(Z)

        num_outputs = self.model.q_sqrt.shape[0]
        self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)))
        self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]))

        self.opt.minimize(self.model.training_loss_closure((X, Y)), self.model.trainable_variables, options=dict(maxiter=iters))

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m
