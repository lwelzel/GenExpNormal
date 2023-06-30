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
        tau (Tensor): Relaxation time parameter of the exponential distribution used for modification.
        validate_args (bool, optional): Whether to enable parameter validation. Default: None.

    Note:
        This is a central distribution as the exponential is not shifted.
        When fitting to a distribution make sure that the 0.5 percentile is at (or near) zero.
    """
    arg_constraints = {'loc': constraints.real,
                       'scale': constraints.positive,
                       'tau': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self,
                 loc: Tensor = torch.tensor([0.]),
                 scale: Tensor = torch.tensor([1.]),
                 tau: Tensor = torch.tensor([0.]),
                 validate_args: bool = None) -> None:

        self.loc, self.scale, self.tau = broadcast_all(loc, scale, tau)
        self.rate = torch.sign(tau) * 1 / (torch.abs(tau) + 1.e-8)
        super().__init__(validate_args=validate_args)

        # get sign of rate to change the modification
        self.sign = torch.sign(self.tau)
        self._tau = torch.abs(self.tau)
        self._rate = torch.abs(self.rate)

        # set all distributions where there should be no exponential modification to a valid rate
        try:
            self._rate[torch.argwhere(self._rate == 0.)] = 1.
        except IndexError:
            if self._rate.item == 0.:
                self._rate = torch.tensor([1.])

        self.normal = Normal(loc=self.loc, scale=self.scale)
        self.gaussian_amplitude = torch.exp(self.normal.log_prob(self.loc))

        self.exponential = Exponential(rate=self._rate)

    def __repr__(self):
        s = f"Exponentially Modified Normal distribution:\n" \
            f"\t LOC:    {self.loc.item():.3e}                \n" \
            f"\t SCALE:  {self.scale.item():.3e}              \n" \
            f"\t TAU:    {self.tau.item():.3e}                \n" \
            f"\t RATE:   {self.rate.item():.3e}               \n" \
            f"\t SIGN:   {self.sign.item():.3e}               \n" \
            f"\t |RATE|: {self._rate.item():.3e}              \n" \
            f"\t N-AMPL: {self.gaussian_amplitude.item():.3e} \n"
        return s

    def sample(self, sample_shape: torch.Size() = torch.Size()) -> Tensor:
        with torch.no_grad():
            return self.sign * (self.normal.sample(sample_shape) + self.exponential.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size() = torch.Size()) -> Tensor:
        return self.sign * (self.normal.rsample(sample_shape) + self.exponential.rsample(sample_shape))

    def pdf(self, value: Tensor) -> Tensor:
        value = self.sign * value

        output = torch.zeros_like(value)

        shift = (value - self.loc) / self.scale
        z = 1 / 1.41421356237 * (self.scale / self._tau - shift)
        # case 0: |z| = 0
        if self.rate == 0.:
            assert len(value.shape) == 1., "Only non batched input for this for now."
            return torch.exp(self.normal.log_prob(self.sign * value))

        # case 1: z < 0
        mask1 = torch.logical_and(z < 0., z >= -6.71e7)
        if torch.sum(mask1) > 0.:
            output[mask1] = self.gaussian_amplitude * self.scale / self._tau * torch.sqrt(torch.pi / torch.tensor([2.])) \
                           * torch.exp(torch.tensor([0.5]) * torch.pow(self.scale / self._tau, 2.)
                                       - (value[mask1] - self.loc) / self._tau) \
                           * torch.erfc(z[mask1])

        # case 2: 0 <= z <= 6.71 x 10^7
        mask2 = torch.logical_and(z >= 0., z <= 6.71e7)
        if torch.sum(mask2) > 0.:
            output[mask2] = self.gaussian_amplitude * self.scale / self._tau * torch.sqrt(torch.pi / torch.tensor([2.])) \
                            * torch.exp(- torch.tensor([0.5]) * torch.pow(shift[mask2], 2.)) \
                            * torch.special.erfcx(z[mask2])

        # case 3: z > 6.71 x 10^7
        mask3 = torch.logical_or(z >= 6.71e7, z <= -6.71e7)
        if torch.sum(mask3) > 0.:
            output[mask3] = self.gaussian_amplitude * torch.exp(- torch.tensor([0.5]) * torch.pow(shift[mask3], 2.)) \
                            / (torch.tensor([1.]) + self._tau * shift[mask3] / self.scale, 2)

        return output

    def _pdf(self, value: Tensor) -> Tensor:  # depreciated
        value = self.sign * value
        return self._rate / 2 \
            * torch.exp(self._rate / 2 * (2 * self.loc + self._rate * self.scale ** 2. - 2. * value)) \
            * torch.erfc((self.loc + self._rate * self.scale ** 2. - value) / (1.41421356237 * self.scale))

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
    tau = torch.tensor([-1.])

    generalized_exponential_normal = ExponentiallyModifiedNormal(loc=loc, scale=scale, tau=tau)
    print(generalized_exponential_normal)

    samples = generalized_exponential_normal.sample((10000, ))

    x_space = torch.linspace(samples.min(), samples.max(), 1000)
    pdf = generalized_exponential_normal.pdf(x_space)
    _pdf = generalized_exponential_normal._pdf(x_space)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 4))

    ax.hist(samples.cpu().numpy().flatten(), bins=50, alpha=0.8, color="red", density=True, label="Samples", zorder=0)

    ax.scatter(x_space[::20], pdf[::20], label="PDF", c="black", s=4, marker="D")
    ax.plot(x_space, pdf, ls="dashed", c="black", lw=0.75, label="PDF")

    ax.set_ylabel("Density")
    ax.set_xlabel(r"$Z_{Gen.~EMG} = X_{\mathcal{N}} + Y_{Exp}$")
    ax.set_title("Generalized Exponentially Modified Normal\n" + r"$\mu=0, ~ \sigma=1, ~ \lambda=-1$")
    ax.legend()
    plt.show()

