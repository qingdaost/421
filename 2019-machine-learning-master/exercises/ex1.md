#  exercise :one:   

1. We looked at regression, and the Gaussian noise model. For 2-class classification, we could push $`\phi`$ through the sigmoid non-linearity as per Friday's lecture, and choose to interpret the output $`y`$ as the _probability of class 1_ compared to class 0, for the current input $`\mathbf{x}`$. So now, instead of the Gaussian relating target and output to each other, we have $`\Pr(correct) = y \; \text{ if t=1}`$ and $`\Pr(correct) = 1-y \; \text{ if t=0}`$. A succinct way to write this is $`\Pr(correct) = y^t (1-y)^{1-t}`$  (check this works for both the cases!). Following the same steps as we did in lecture, show that gradient ascent of the log of this likelihood gives the perceptron learning rule.


2. Back to the linear + Gaussian case: suppose we think the noise model is still Gaussian, but has a variance that is considerably smaller than 1. What effect will this have on the log likelihood, and the learning rule?  _Note: the (1-dimensional) Gaussian distribution having mean $`\mu`$ and variance $`\sigma^2`$ is 
$`\Pr(z) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\big[- \frac{(z-\mu)^2}{2 \sigma^2} \big]`$. In our case, the mean of the noise is zero, so just pay attention to the effect of $`\sigma^2`$ no longer being 1._


