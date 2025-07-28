## Disentanglement and Interpretability of StyleGAN

StyleGAN is a GAN (_Generative Adversarial Network_) architecture.

Disentanglement here refers to the degree that individual latent coordinates or directions correspond to single interpretable attributes, without entangling multiple attributes. Recent work (Dinh et al., 2022) directly measured disentanglement in StyleGAN3 using the DCI metric (Disentanglement, Completeness, Informativeness). In fact, StyleGAN3’s $W$ space is more entangled than StyleGAN2’s (for aligned face models).

[Belanec et al. (2024) Controlling the Output of a Generative Model by Latent Feature Vector Shifting](https://arxiv.org/pdf/2311.08850v2)

### Feature extraction :

[Nitzan et al. (2020) Face Identity Disentanglement via Latent Space Mapping](https://arxiv.org/abs/2005.07728)

[Shawky Sabae et al. (2022)](https://arxiv.org/pdf/2204.07924) see page 4

[Closed-Form Factorization of Latent Semantics in GANs (2020)](https://arxiv.org/abs/2007.06600)

### Feature interpolation :

https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network
