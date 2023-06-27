# GenExpNormal
Generalized Exponentially modified Normal distribution in PyTorch (Exponentially Modified Gaussian distribution [EMG], ExGaussian). [See EMG on Wikipedia](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution) for more details on the non general version (right skew).

![GenNormExp-example](https://github.com/lwelzel/GenExpNormal/assets/29613344/c46d8d63-d068-4969-bea0-9198c0475787)

Exponentially modified Normal distribution based on PyTorch Distribution and intended to integrate with the PyTorch framework. The exponentially modified normal is generalized to also accept negative or zero rate which skews the distribution left instead of right. Batch sampling is implemented for mixed positive and negative rate distributions  and fully normal ones (rate = 0) via parameter tensors.
