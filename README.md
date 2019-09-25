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
               
## BatchNorm2d : 
- Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
- torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 
               
               - num_features: CCC from an expected input of size (N,C,H,W)(N, C, H, W)(N,C,H,W)
               - eps: a value added to the denominator for numerical stability. Default: 1e-5
               - momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
               - affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
               - track_running_stats: a boolean value that when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: True
               
## LeakyReLU:
- torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
                        
                - LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
                - negative_slope – Controls the angle of the negative slope. Default: 1e-2
                - inplace – can optionally do the operation in-place. Default: False



               
