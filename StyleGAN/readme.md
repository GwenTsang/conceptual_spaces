## Disentanglement and Interpretability of StyleGAN

StyleGAN is a GAN (_Generative Adversarial Network_) architecture.

Disentanglement here refers to the degree that individual latent coordinates or directions correspond to single interpretable attributes, without entangling multiple attributes. Recent work (Dinh et al., 2022) directly measured disentanglement in StyleGAN3 using the DCI metric (Disentanglement, Completeness, Informativeness). In fact, StyleGAN3’s $W$ space is more entangled than StyleGAN2’s. For aligned face models, StyleGAN2’s $W$ had a Disentanglement of 0.54, whereas StyleGAN3’s (aligned) $W$ was 0.47; the Completeness similarly dropped from 0.57 to 0.43. Notably, StyleGAN3’s $S$ space actually scores higher (Disent. 0.89) than StyleGAN2’s (0.75). This suggests that while StyleGAN3’s $W/W+$ lost some linear disentanglement relative to StyleGAN2, its $S$ space remains highly disentangled.


### Feature extraction :

[Nitzan et al. (2020) Face Identity Disentanglement via Latent Space Mapping](https://arxiv.org/abs/2005.07728)

[Shawky Sabae et al. (2022)](https://arxiv.org/pdf/2204.07924) see page 4

[Closed-Form Factorization of Latent Semantics in GANs (2020)](https://arxiv.org/abs/2007.06600)

### Feature interpolation :

https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network
