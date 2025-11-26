import jax.random as jr
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from dynamax.linear_gaussian_ssm import lgssm_filter
from dynamax.linear_gaussian_ssm import LinearGaussianSSM as dmax_LGSSM

from feynmac_kac.protocol import FeynmacKac, PFConfig


class LGSSMModel(FeynmacKac):
    def __init__(self, params):
        super().__init__(params)

    def p0(self, key, N):
        m1 = self.params['m1']
        Q1 = self.params['Q1']
        return jr.multivariate_normal(key, m1, Q1, shape=(N, ))

    def pt(self, key, x_prev, t):
        A = self.params['A']
        b = self.params['b']
        Q = self.params['Q']
        mean = A @ x_prev + b
        return jr.multivariate_normal(key, mean, Q)
    
    def beta_pt(self, key, x_prev, t, beta):
        A = self.params['A']
        b = self.params['b']
        Q = self.params['Q']
        mean = A @ x_prev + b
        return jr.multivariate_normal(key, mean, Q / jnp.maximum(beta, 0.1))

    def log_g(self, t, x_t, x_prev, y_t):
        # C = self.params['C'][0]  # single modality for now
        # # d = self.params['d'][0]
        # R = self.params['R']
        # mean = C @ x_t
        # diff = y_t - mean
        # exponent = -0.5 * diff.T @ jnp.linalg.inv(R) @ diff
        # norm_const = -0.5 * (jnp.log(jnp.linalg.det(2 * jnp.pi * R)))
        # return norm_const + exponent
        C = self.params['C']
        R = self.params['R']
        mean = C @ x_t
        return multivariate_normal.logpdf(y_t, mean=mean, cov=R)
    
    def log_f(self, t, x_t, y_t):
        return self.log_g(t, x_t, jnp.array([]), y_t)
    
    def log_pt(self, t, x_t, x_prev):
        A = self.params['A']
        b = self.params['b']
        mean = A @ x_prev + b
        return multivariate_normal.logpdf(x_t, mean=mean, cov=self.params['Q'])
    
    def log_p0(self, x_0):
        m1 = self.params["m1"]
        Q1 = self.params["Q1"]
        return multivariate_normal.logpdf(x_0, mean=m1, cov=Q1)

def main():

    import matplotlib.pyplot as plt

    from feynmac_kac.bootsrap_2 import BPF
    from utils.datasets import load_dataset
    
    data = load_dataset('linear_small', seed=0)

    # get true filter means using Kalman filter
    dmax_lgssm = dmax_LGSSM(data.train_states.shape[-1], data.train_data[0].shape[-1])
    params, _ = dmax_lgssm.initialize(
        jr.PRNGKey(0),
        initial_mean=data.params['m1'],
        initial_covariance=data.params['Q1'],
        dynamics_weights=data.params['A'],
        dynamics_covariance=data.params['Q'],
        emission_weights=data.params['C'],
        emission_covariance=data.params['R']
    )
    x, y = data.train_states[0], data.train_data[0][0]
    lgssm_posterior = lgssm_filter(params, y)
    print("Exact filter means shape: ", lgssm_posterior.filtered_means.shape)

    # parametrise model with the true parameters for now
    lgssm = LGSSMModel(data.params)

    # construct bootstrap particle filter
    bpf = BPF(
        model=lgssm,
        config=PFConfig(
            N=100,
            resample_scheme='multinomial',
            ess_threshold=0.5,
            key=jr.PRNGKey(42)
        )
    )

    # run filter
    outs = bpf.filter(
        T=data.train_data[0].shape[1],
        obs=data.train_data[0][0]  # just using the first sequence
    )
    print("BPF particles shape: ", outs.particles.shape)
    print("BPF weights shape: ", outs.weights.shape)
    print("BPF logZ_hat: ", outs.logZ_hat)

    # calculate weighted estimates of state means
    x_hat = jnp.sum(outs.weights[:, :, None] * outs.particles, axis=1)
    print("BPF weighted x_t shape: ", x_hat.shape)

    # calculate MSE against true filter means
    mse = jnp.mean((x_hat - lgssm_posterior.filtered_means)**2)
    print("MSE from BPF to exact filter means: ", mse)

    # plot results
    observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(y, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
    ax.plot(x, ls="--", color="darkgrey", label="true state")
    ax.plot(lgssm_posterior.filtered_means, color="blue", label="filtered mean")
    ax.plot(x_hat, color="tab:red", label="BPF estimate")

if __name__ == "__main__":
    main()



