# optimisation and regularisation (focussing on neural nets)

## Lecture :eight: (Thursday, week 3) - Optimisation (_Deep Learning_, Chapter 8)

This material is (nicely ?) covered in [Chapter 8 of _Deep Learning_](https://www.deeplearningbook.org/contents/optimization.html), but that goes into more mathematical depth than we need here. (Some - for example _conjugate gradients_ - the following is covered in Marsland, Chapter 9, but there is lots here that's not in that Chapter, and vice versa).

We're only covering the points made below, which is all in the _Deep Learning_ chapter. There's a good looking [blog here](https://blog.algorithmia.com/introduction-to-optimizers/) too.

We have seen how the gradient can be calculated, ie. we know the gradient of loss of some learner, with respect to each of its learnable parameters. Surely this is useful for optimization...


#### character of the loss surface
 * In training a neural net we are doing _non-convex optimization_. Even convex optimization is typically very hard to do.
 * local minima
 * symmetries in weight space - we say the model is _not identifiable_
 * in many classes of random function: in low-dim spaces, local minima are common. In high-dim spaces, local minima rare but saddle points common: expected ratio of saddle points to local optima grows _exponentially with $`n`$. (Nb. Hessian $`\mathbf{A}`$ at a local opt has only positive eigenvalues, while at a saddle there is a mix of positive and negative. If sign of eigenvalue were a tossed coin... QED).
 * saddle points - now recognised as very prevalent in neural net loss surfaces. Would be major problem for Newton's alg, and _could_ be problem for gradient descent.

Cliffs and plateaus = exploding and vanishing gradients:
 * Eigenvalues of the product of several weights matrices (even with no non-linearities!).
 * most strikingly, in recurrent neural nets (coming later in the course)
 * exploding gradients make cliffs
 * vanishing gradients make plateaus - obviously a problem for gradient descent. NB squashing functions as non-linearities tend to generate these regions even more.

Gradients in deep neural networks are also *shattered*, _especially_ when using Batch Norm (discussed below). See the poster outside my office :grin: or [our ICML paper](http://proceedings.mlr.press/v70/balduzzi17b.html) if you're interested. Completely separate problem from vanishing/exploding gradients.

#### Gradient Descent: is it even a good idea?
 * $`\delta w_i = -\eta \nabla_w \frac{\partial \mathcal{L}}{\partial w_i}`$ or in vector notation $`\delta \mathbf{w} = -\eta \nabla_\mathbf{w} \mathcal{L}`$ where $` \nabla_\mathbf{w} \equiv \partial/\partial\mathbf{w}`$
 * how to set the learning rate? Problems with being too small/big.
 * dimensionally inconsistent - (details in 34.3 of David MacKay's book [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/mackay/itila/p0.html#book.html) )
   * there is a good argument for pre-multiplying by the _inverse of the Hessian_ : $`\mathbf{M} = \mathbf{A}^{-1}`$ where $` \mathbf{A} \equiv - \nabla_\mathbf{w} \nabla_\mathbf{w} \mathcal{L} \mathcal{L}`$, but unlike the gradient that's expensive to calculate, and very sensitive to minibatches (see below)

 
#### minibatchs / "stochastic gradient descent" or just "SGD"
 * batch or deterministic gradient descent: average the gradient over the entire training set, before taking a step. That's pretty wasteful - imagine a data set with lots of near-repetitions in it...
 * one-at-a-time:  take a (small) step after every single item in the training set (you probably want to go through it in a randomized order in that case). 
 * minibatch or stochastic: use a subset of data to compute a gradient and take a step. And work your way through the whole of the data this way.
 * the "noise" in minibatches can help optimization
 * size of minibatches? It's a "hyper-parameter", best set by magic / trial and error.
 
 
#### Momentum
 * do SGD, but add in "a bit" of the last gradient as well. Very widely used.
 * let's SGD pick up speed on long slopes...
 * ... and dampens its oscillations in narrow ravines.
 * typical value for "a bit" is $`\alpha = 0.9`$
 * Q: what is "Nesterov momentum"? A: a horrible hack. May sometimes work...
  
### A. adaptive learning rates
 * heuristic ideas, eg:
   * "bold driver": if the loss went down, double the learning rate, and if it went *up* then halve it. Yep. 
   * "delta-bar-delta": if gradient vector has positive dot product with the previous one, increase the learning rate. Yep.
   
##### AdaGrad 
scale the learning rate as you go, making it proportional to inverse of the sum of historical squared values of the gradient: 

Suppose minibatch has gradient vector (one for each weight), $`\mathbf{g}`$ (ie. the mean over the minibatch of the gradient of the loss function).

We keep a vector of parameters  (one for each weight), $`\mathbf{r}`$, that will determine learning rates of each weight.

Update those:  $`\mathbf{r} \longleftarrow \mathbf{r} + \mathbf{g} \odot \mathbf{g}`$, where $`\odot`$ is element-wise multiplication, a.k.a. "Hadamard product".

Update weights: $`\Delta \mathbf{w} \longleftarrow - \frac{\epsilon}{\delta+\sqrt{r}} \odot \mathbf{g}`$

(with everything being done element-wise. $`\epsilon`$ is the global learning rate. $`\delta`$ is just to stabilize division by small numbers).

Nice properties in convex situations, but its "long memory" makes for problems in the non-convex setting.

##### RMSprop (Hinton 2012) 

Similar to AdaGrad, but we fade out older contributions to $`\mathbf{r}`$. Degree of fading is a parameter, $`0 < \rho < 1`$.

Update those rate parameters:  
$`\mathbf{r} \longleftarrow \rho \mathbf{r} + (1-\rho) \mathbf{g} \odot \mathbf{g}`$.

Empirically: works well in deep learning - one of the go-to methods.

(People sometimes combine it with Nesterov momentum).

##### Adam and Nadam

* Adam is like RMSprop plus momentum, with subtle distinctions. No theory, but works :|
* Nadam is Adam with Nesterov momentum thus one hack after another :(

![alec radford demo](https://d3ansictanv2wj.cloudfront.net/image1-44043daf8f4edbaf8e0d669c83be593c.gif)
![sebastian ruder demo](http://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif)
![another ruder](http://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif)

* Nice looking blog post ["10 grad descent algs"](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)

* Even nicer looking blog post from [Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/)


### B. Approximate 2nd order methods

##### Newton's method

 * Idea: approximate the local loss (error) surface as a _quadratic bowl_, and simply _jump to the bottom of that bowl_ in one step. If it's not really quadratic, you need to iterate.
 * Could be really fast - big jumps! If the surface really is quadratic, you're all done in one iteration!
 * But:
    * as well as gradients, we need to find the second derivatives, and then do a matrix inverse (ie. "calculate the inverse Hessian") to get the parabola's lowest point.
    * but non-convexity can cause severe problems (especially saddle points: it's NOT actually a parabolic bowl!).
    * NOISE (due to minibatches) causes problems too.
    * EXPENSIVE: if there are k weights, the Hessian has k*k numbers to be found.
 * a low-memory approximation called L-BFGS exists, and works. It's not that widely used - not sure why.


<!--
I ran out of time in this lecture, for Conj Grad, and Batch Norm.
Deciding (now) to clip out Conj Grad altogether. A pity but oh well!

##### Conjugate gradients
In outline: 

* Consider the "line search" algorithm: find the gradient at the current point in weight space, and then do a full search along that direction for the lowest point. This is relatively do-able (essentially, 1-dim search is reliable and fast). Jump to that point, and iterate...
* As a heuristic, this seems sort of sensible.
* PROBLEM: line search's second direction *will be* orthogonal to the first one (and so on). This causes zig-zagging, which slows convergence.
* loosely: uses (approximate) second-order information, without computing the Hessian
* solves the zig-zagging problem that _line search_ would have
* not that widely used - again, not sure why. Perhaps it's that line search itself isn't so great as a heuristic. 
-->

### C. Batch normalisation

 * The issue: "covariate shift" - there is strong coupling between learning at different layers, in "vanilla" NN learning.

"Batch Norm" (2015) attempts to ameliorate this problem.

Consider a batch of data, and a neuron somewhere in a neural net. This sees a _set_ $`\{\phi\}`$ of values for the weighted sum.
We could deliberately _shift and rescale_ these so that they are a "normalised" distribution : have mean zero and variance 1. That is: we find the actual mean, and subtract it, and the actual standard deviation, and divide by it. 

But the net might prefer to use other values (e.g. it's not necessarily ideal for a ReLU to output zero exactly half the time...). So we can have parameters giving a particular mean $`\beta`$ and standard deviation $`\gamma`$.  

When learning, we treat the normalisation as just another function that $`\phi`$ has been through, ie. we 'backprop _through_ the normalisation'. And we can learn parameters $`\beta,\gamma`$.  

See [Andrew Ng's video explanation](https://www.youtube.com/watch?v=em6dfRxYkYU) (especially the first few minutes) - note that his nomenclature is $`z`$ for the weighted sum into a unit (what we called $`\phi`$) and $`a`$ for activation of a unit (what we called its output, $`h`$).

What's it doing?!
 * limits the amount that updates to weights in early layers affect the distribution of values in later layers
 * [Andrew Ng talks about 'why it works'](https://www.youtube.com/watch?v=nUUqwaxLnWs)
 * even though the input distribution (to say layer 3) changes in response to changes (at layer 1), it changes less"
 * Nb. also has a slight 'regularisation' effect - let's revisit that tomorrow though.

 



## Lecture :nine: (Friday, week 3)  - Regularisation (_Deep Learning_, Chapter 7)

We first went back to complete the previous lecture, ie. Batch Normalisation. SEE ABOVE.

Then: regularisation is "complexity control". We have talked about training aimed at getting a training set correct, but it's the test error that we should really care about. Regularisation refers to efforts to reduce the _test_ error, even if that means sacrificing some _training_ error.

[Chapter 7 of _Deep Learning_](https://www.deeplearningbook.org/contents/regularization.html) (Goodfellow et al.) covers all this, but we will not go to the same mathematical depth at all. Instead, we will summarise the main ideas and some current techniques.

Background notions:
 * No Free Lunch theorem: generalisation only happens if our model (somehow) builds in assumptions that happen to apply in the real world.
 * Ockham's razor: if two models both fit the same data, the simpler one is more likely to generalise well.

Some kinds of regularisation routinely employed in deep learning:
 * L2 regularisation
   * what it is
   * how it causes "weight decay" in the learning rule
   * how should we set the amount?
 * Earlier, we motivated the Loss Function via a probabilistic interpretation: the _likelihood_ is a big product (i.i.d. assumption), and max-ing the log likelihood gives a sum over the training set. An additive penalty is ... what? 
   * additive "penalty" term is a multiplier, before going through the logarithm. We can interpret as the consequence of having a _prior_, and doing MAP inference instead of max likelihood. Bayesian view: we find the most plausible parameters, not those most likely to get the training set correct.
 * L1 regularisation and sparsity. _Cf_ L0.
 
 * Augmenting the data set is a kind of regularisation
 * Injecting noise into target values is another
 * Early stopping is another
 * Weight sharing
 * Dropout
 * Batch Norm?
 * adversarial training

