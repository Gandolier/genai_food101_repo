0) Initial model that we won't pay attention to since it was created just to check whether the pipeline works.

1) Main DDPM-architecture model that we will base upon.

2) Changed the learning rate based on the error performance during training process. Increased the initial learning rate from 10^{-4} to 10^{-3} and the minimal learning rate from 10^{-6} to 10^{-5}. Model is still inadequate.

3) Decreased batchsize from 64 to 16. That implies that the singe training epoch now takes ~101000/16=6312 steps. Now, if we assume one epoch (that is 6312 steps) to be the wavelength of our cosine scheduler and that scheduler will make 30 steps to oscillate one wavelength we will need to update our learning rate according to scheduler every ~6312/30=210 steps, that also implies T_max=105. The size of the timesteps of noise generation was also changed from 100 to 500. Also, size of the time_embedding was increased from 128 to 256.

