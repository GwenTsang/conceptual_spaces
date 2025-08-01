## Disentanglement and Interpretability of StyleGAN

StyleGAN is a GAN (_Generative Adversarial Network_) architecture.

The file `smile_direction.npy` is an initial test of smile property extraction in W+ latent space. However, the data from which we extracted this vector appears to contain a large number of false positives.

Install StyleGAN3 specialized in human faces aligned in 1024x1024 pixel format  :

```
!git clone https://github.com/NVlabs/stylegan3.git
!wget -nc -O stylegan3/stylegan3-t-ffhq-1024x1024.pkl https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl
```

Disentanglement here refers to the degree that individual latent coordinates or directions correspond to single interpretable attributes, without entangling multiple attributes. Recent work (Dinh et al., 2022) directly measured disentanglement in StyleGAN3 using the DCI metric (Disentanglement, Completeness, Informativeness).

[Belanec et al. (2024) Controlling the Output of a Generative Model by Latent Feature Vector Shifting](https://arxiv.org/pdf/2311.08850v2)

### Feature extraction :

[Nitzan et al. (2020) Face Identity Disentanglement via Latent Space Mapping](https://arxiv.org/abs/2005.07728)

[Shawky Sabae et al. (2022)](https://arxiv.org/pdf/2204.07924) see page 4

[Shen et al. (2020) Closed-Form Factorization of Latent Semantics in GANs](https://arxiv.org/abs/2007.06600)

### Feature interpolation :

https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network
