# Generative-adversarial-network-GAN-
The main focus for GAN (Generative Adversarial Networks) is to generate data from scratch, mostly images but other domains including music have been done. But the scope of application is far bigger than this. Just like the example below, it generates a zebra from a horse. In reinforcement learning, it helps a robot to learn much faster.

- GAN composes of two deep networks, the generator, and the discriminator. 
- Let x be data representing an image. D(x) is the discriminator network which outputs the (scalar) probability that x came from training data rather than the generator. Here, since we are dealing with images the input to D(x) is an image of CHW size 3x64x64. Intuitively, D(x) should be HIGH when x comes from training data and LOW when x comes from the generator. D(x) can also be thought of as a traditional binary classifier.

- D(G(z)) is the probability (scalar) that the output of the generator G is a real image. As described in Goodfellow’s paper, D and G play a minimax game in which D tries to maximize the probability it correctly classifies reals and fakes (logD(x)), and G tries to minimize the probability that D will predict its outputs are fake (log(1−D(G(x)))). From the paper, the GAN loss function is
               
                        - minGmaxDV(D,G)=Ex∼pdata(x)[logD(x)]+Ez∼pz(z)[log(1−D(G(z)))]
                        - lassifies reals and fakes (logD(x))
                        - D will predict its outputs are fake (log(1−D(G(x))))
      
     
## DC GAN :
- A DCGAN is a direct extension of the GAN described above, except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator
- The discriminator is made up of strided
                           - convolution layers
                           - batch norm layers
                           - LeakyReLU activations.
                
## convolution layers : 
- torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,      dilation=1, groups=1, bias=True, padding_mode='zeros')

               - stride: controls the stride for the cross-correlation, a single number or a tuple.
               - padding: controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension.
