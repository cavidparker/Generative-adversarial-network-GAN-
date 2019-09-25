# Generative-adversarial-network-GAN-
The main focus for GAN (Generative Adversarial Networks) is to generate data from scratch, mostly images but other domains including music have been done. But the scope of application is far bigger than this. Just like the example below, it generates a zebra from a horse. In reinforcement learning, it helps a robot to learn much faster.

- GAN composes of two deep networks, the generator, and the discriminator. 
- Let x be data representing an image. D(x) is the discriminator network which outputs the (scalar) probability that x came from training data rather than the generator. Here, since we are dealing with images the input to D(x) is an image of CHW size 3x64x64. Intuitively, D(x) should be HIGH when x comes from training data and LOW when x comes from the generator. D(x) can also be thought of as a traditional binary classifier.
