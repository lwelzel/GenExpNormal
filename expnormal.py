import torch
from torch.distributions import constraints, Distribution, Exponential,  Normal
from torch import Tensor
from torch.distributions.utils import broadcast_all

__all__ = ['ExponentiallyModifiedNormal']

class ExponentiallyModifiedNormal(Distribution):
    """
    Exponentially Modified Normal distribution.

    Args:
        loc (Tensor): Mean of the normal distribution.
        scale (Tensor): Standard deviation of the normal distribution.
        rate (Tensor): Rate parameter of the exponential distribution used for modification.
        validate_args (bool, optional): Whether to enable parameter validation. Default: None.

    Note:
        This is a central distribution as the exponential is not shifted.
        When fitting to a distribution make sure that the 0.5 percentile is at (or near) zero.
    """
    arg_constraints = {'loc': constraints.real,
                       'scale': constraints.positive,
                       'rate': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self,
                 loc: Tensor = torch.tensor([0.]),
                 scale: Tensor = torch.tensor([1.]),
                 rate: Tensor = torch.tensor([0.]),
                 validate_args: bool = None) -> None:

        self.loc, self.scale, self.rate = broadcast_all(loc, scale, rate)
        super().__init__(validate_args=validate_args)

        # get sign of rate to change the modification
        self.sign = torch.sign(self.rate)

        self._rate = torch.abs(self.rate)

        # set all distributions where there should be no exponential modification to a valid rate
        try:
            self._rate[torch.argwhere(self._rate == 0.)] = 1.
        except IndexError:
            if self._rate == 0.:
                self._rate = 1.

        self.normal = Normal(loc=self.loc, scale=self.scale)

        self.exponential = Exponential(rate=self._rate)

    def sample(self, sample_shape: torch.Size() = torch.Size()) -> Tensor:
        with torch.no_grad():
            return self.sign * (self.normal.sample(sample_shape) + self.exponential.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size() = torch.Size()) -> Tensor:
        return self.sign * (self.normal.rsample(sample_shape) + self.exponential.rsample(sample_shape))

    def pdf(self, value: Tensor) -> Tensor:
        return self._rate / 2 \
            * torch.exp(self._rate / 2 * (2 * self.loc + self._rate * self.scale ** 2. - 2. * self.sign * value)) \
            * torch.erfc((self.loc + self._rate * self.scale ** 2. - self.sign * value) / (1.41421356237 * self.scale))

    def log_prob(self, value: Tensor) -> Tensor:
        return torch.log(self.pdf(value))

    # TODO: implement @property mean, mode, stddev. variance:
    #  https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def mode(self) -> Tensor:
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # define parameters, note that the rate can be negative
    # batch sampling is implemented for mixed positive and negative rate distributions and fully normal ones (rate = 0)
    loc = torch.tensor([0.])
    scale = torch.tensor([1.])
    rate = torch.tensor([-1.])

    generalized_normal = ExponentiallyModifiedNormal(loc=loc, scale=scale, rate=rate)

    samples = generalized_normal.sample((10000, ))

    x_space = torch.linspace(-10, 5, 100)
    pdf = generalized_normal.pdf(x_space)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 4))

    ax.hist(samples.cpu().numpy().flatten(), bins=50, alpha=0.8, color="red", density=True, label="Samples", zorder=0)

    ax.scatter(x_space[30:80:2], pdf[30:80:2], label="PDF", c="black", s=4, marker="D")
    ax.plot(x_space, pdf, ls="dashed", c="black", lw=0.75)

    ax.set_ylabel("Density")
    ax.set_xlabel(r"$Z_{Gen.~EMG} = X_{\mathcal{N}} + Y_{Exp}$")
    ax.set_title("Generalized Exponentially Modified Normal\n" + r"$\mu=0, ~ \sigma=1, ~ \lambda=-1$")
    ax.legend()
    plt.show()

