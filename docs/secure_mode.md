# Opacus' Secure Mode

Part of the process for achieving a differential privacy guarantee under Opacus involves generating noise according to a Gaussian distribution with mean 0 in Opacus' `_generate_noise()` function.

Enabling `secure_mode` when using the NHSSynth package ensures that the generated noise is also secure against floating point representation attacks, such as the ones in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.

This attack first appeared in https://arxiv.org/abs/2112.05307; the fix via the [`csprng`](https://github.com/pytorch/csprng) package is based on https://arxiv.org/abs/2107.10138 and involves calling the Gaussian noise function $2n$ times, where $n=2$ (see section 5.1 in https://arxiv.org/abs/2107.10138).

The reason for choosing $n=2$ is that $n$ can be *any* number greater than $1$. The bigger $n$ is, though, the more computation needs to be done to generate the Gaussian samples. The choice of $n=2$ is justified via the knowledge that the attack has a complexity of $2^{p(2n-1)}$. In PyTorch, $p=53$ and so the complexity is $2^159$, which is deemed sufficiently hard for an attacker to break.