# preliminaries

Laying down some basic understandings.

##  lecture :one: 

#### course stuff
 * The Plan: largely neural nets until the "break", and largely probabilistic stuff after then. But the split isn't clean at all - there is lots of cross-connection.
 * Marcus lectures the first half; Bastiaan the second
 * [Marsland](https://tewaharoa.victoria.ac.nz/primo-explore/fulldisplay?docid=TN_pq_ebook_centralEBC1591570&context=PC&vid=VUWNUI&lang=en_NZ&search_scope=64VUW_ALL&adaptor=primo_central_multiple_fe&tab=all&query=any,contains,Marsland%20Machine%20Learning&facet=rtype,exclude,newspaper_articles&facet=rtype,exclude,reviews)
 the main text for first half; [Barber](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage)
 the second
 * We will also make occasional use of [Goodfellow et al.](http://www.deeplearningbook.org/) 


#### first comments on machine learning
 * data is a big matrix of inputs X mapping (in "supervised learning" at least) to a matrix of "targets" Y
 * When Y real the task is called "regression", Y mutually exclusive options is called "classification".
 * we try to learn a mapping that takes X to Y
 * this is done in the expectation that a good mapping will yield a sensible result y when given a novel x. Remember it's the novel one that actually counts!
 * two broad options for the mapping: non-parametric (e.g. k-Nearest-Neighbours) and parametric (e.g. fitting a line, or a perceptron).
 * On the face of it, non-parametric methods invite trouble from the Curse of Dimensionality, namely that in high dimensional spaces, the relative volume of hypersphere vs hypercube is miniscule: almost all the volume is "in the corners"


##  lecture :two:  (Thurs of week 1)

 * two other things true of high dimensional spaces:
  * random vectors are orthogonal
  * typical distance between any two vectors tends to be the same (try it out!)
 * we will come across numerous consequences for machine learning, e.g. kNN says nearest point is privileged, but all points are becoming equally "near"!
 * COMMENT: _using_ a non-parametric method may be costly, whereas _learning_ a parametric method may be costly.
 * one of the simplest possible parametric classifiers: form a linear weighted combination of the $`x`$ values (the input), ie. $`\phi=\mathbf{x}\cdot\mathbf{w} + w_{bias}`$. Then (optionally) put that "through" a simple non-linearity, namely the threshold function $`y = 1 \text{ if } \phi \geq 0`$ and 0 otherwise. This is called a linear classifier or perceptron (a term from the 1950s), and _can_ be seen as a very crude model of what biological neurons do.
 * this connection (to brains) is more coincidental / motivational than anything else. It's true that some researchers have been heavily influenced by real brains - after all they are the "existence proofs" that machine learning of the real world might even be possible.
 * all the points (let's call them $`\hat\mathbf{x}`$) on the "decision boundary" of a perceptron satisfy $` \hat\mathbf{x} \cdot \mathbf{w} + w_{bias} = 0`$, which implies the decision boundary is a _hyperplane_ and _perpendicular to the weights vector_.
 * Please read Marsland Chapters 2 and 3.
 * The Perceptron Learning Rule $`\Delta w_i = \eta (t-y) x_i `$ (or in vector form $`\Delta \mathbf{w} = \eta (t-y) \mathbf{x} `$), when applied sequentially to each input-target pair $`(\mathbf{x}, t)`$ in the dataset often enough, is guaranteed to converge to a solution in which the perceptron makes zero errors, _if such a solution exists_.


## Lecture :three: (Friday, week 1)

#### perceptron recap
* find a linear weighted sum plus bias, let's call it $`\phi`$. Push it through a "step function" non-linearity to get output $`y`$: outputs and targets are binary, so it's a 2-class classifier.
* The decision surface is a hyperplane, and a data set with 2-class targets (e.g. 0 and 1) which can be split by a hyperplane is called "linearly separable"
* the iconic example of _non-separable_ data is XOR (4 patterns, no possible hyperplane) - see _Marsland_ chapter 3.
* The Perceptron Learning Rule (a.k.a. Delta rule) changes the weights under the influence of data. Remarkably, this rule is _guaranteed to converge_ for linearly separable data (and otherwise it's guaranteed *not* to converge). The learning rate doesn't actually matter, so long as it's positive.
* the convergence proof (not done in class) is 'stand alone' - it doesn't rely on gradient descent (up next).

#### Gradient descent
* basic idea: write down a function of parameters $`\mathbf{w}`$ (and the training data of course) which is zero if the neuron does what we want it to, and positive otherwise. Call this the _loss_, $`\mathcal{L}(\mathbf{w})`$. The loss forms a _surface in weight space_.
* start parameters randomly, then try to improve them by _going downhill on the loss surface_.
* how big a step to take? Yes, that's a good question (which we will fudge for now) - "pick a number that works okay" :grimace:

#### Gradient descent for a linear neuron (a.k.a. "linear regression")
* Suppose $`y=\phi`$ (ie. no non-linearity at all)
* Consider the "squared error" loss $`\mathcal{L}(\mathbf{w}) = \sum_n \frac{1}{2} (t_n - y_n)^2 `$
  * zero _iff_ all the targets are matched exactly, and otherwise positive
* we took the gradient and saw that this leads to the very same delta rule as before.
* that's quite surprising in a way! Totally different setups...
* but why the sum of squared errors instead of something else? Hold that thought.

#### Gradient descent of squared errors, but for a non-linear neuron
* consider the following "squashing function", which maps the real number line onto the interval [0,1], and is often called the "sigmoid" or "logistic" curve:  $`y = \frac{1}{1+e^{-\phi}} `$
* everything's the same, except this time we got a $`\frac{\partial y}{\partial \phi} `$, and that turns out (details avoided, as a bit tedious) to be $`y(1-y)`$, a parabola, peaks at 0.5
* so we get the same old delta rule except uglified a little bit by this multiplier. Fine... except _why the sum of squared errors instead of something else?_


#### Step back: a way to _derive_ a sensible loss function
* SUPPOSE WE THINK that data targets were generated by a process in which a (unknown) function was applied to the input, _and then Gaussian noise corrupted the result_ so $`t = f(\mathbf{x}) + \epsilon`$ where $`\epsilon \sim \mathcal{N}(0,1)`$. 
* What's the probabity that our predictor $`y`$ gets the target $`t`$ correct, for every case in the training set? It would be the probability of getting the first one right, and the second, and...
* since these are _independent_ the whole probability (of getting the whole lot correct) is just a product of the individual ones:
$`\Pr(\text{all correct}) = \prod_n  \Pr(y_n \;\text{matches}\; t_n)`$
* and each individual one is a Gaussian:
$`\Pr(y_n \;\text{matches}\; t_n) =  \frac{1}{\sqrt{2\pi}} \exp(-\frac{(t_n - y_n)^2)}{2}`$
* actually, that's a probability _density_, but that's not a show-stopper.
* maximizing that product is achieved by maximizing its logarithm (which is numerically preferable), which is a sum of logs:
$`\log \Pr(y_n \;\text{matches}\; t_n) = \text{constant} -\frac{1}{2}(t_n - y_n)^2)`$
* The sum of squares seems a reasonably well-motivated loss function to use, if you think the targets have Gaussian noise on them.
* Generalising: the basic idea was to use the logarithm of the likelihood, under the i.i.d. assumption and with a sensible choice of "noise model" (here, Gaussian).


*TUTORIAL QUESTIONS*: see Exercise 1 in [Exercises](../exercises). Write your answers out and bring to Tuesday's session, where we will mark them.

## Lecture :four: (Tuesday, week 2)

We went over the two exercises.

And then reviewed the last section in Marsland Chapter 3, about the bias-variance tradeoff. (note there's a [typo](../MLA-typos.md) there).
TODO: summary of bias-variance here.

## Lecture :five: (Thursday, week 2)

#### Cross-entropy loss for softmax "neural" classifiers

Suppose we have data consisting of a set of $`N`$ patterns $`\mathbf{x}`$ which are $`D`$ elements long, each with a target class $`\in \{1..K\}`$. We can think of this as a $`N\times D`$ matrix $`X`$, and a $`N\times K`$ matrix $`T`$. each row of $`T`$ contains zeros except from a single 1 in the column corresponding to the correct target class.Consider a feed-forward neural network. When the network is given the input vector $` \mathbf{x}{n} `$ it generates an output vector $`\mathbf{y}{n}`$ via the softmax function.
We can make an $`N\times K`$ matrix $`Y`$ from these output vectors (one per row), and thus the network maps $`X \rightarrow Y`$ and $`Y`$ has the same dimensions as $`T`$. The softmax function is

$` Y_{n,i} = \frac{\exp(\phi_{n,i})}{Z_n}`$

where $`\phi_{n,i} = \sum_j W_{i,j} X_{n,j}`$ and $`Z_n = \sum_k \exp(\phi_{n,k})`$.In lecture we met the "cross entropy" loss function:

$` E = \sum_n \sum_k T_{n,k} \log Y_{n,k} `$  

(Note that sometimes it's more convenient to write this as

$` E = \sum_n \log Y_{n, c_n} `$


where $` c_n `$ is the index of the target class for the $` n^\text{th} `$ item in the training set.)We motivated the cross-entropy loss by arguing that it is the log likelihood, ie. the probability that a stochastic form of this network would generate precisely the training classes, namely to use the softmax outputs as a categorical distribution and sample classes from it.(See  this
which discusses cross-entropy _vs_ training error _vs_ sum-of-squared errors without refering to a stochastic model).


Q: Consider a simple neural network with no hidden layers and a softmax as the output layer. Show mathematically that gradient descent of the cross entropy loss in leads to the "delta rule" for the weight change: $`\Delta W_{ij} \propto \sum_n (T_{n,i} - Y_{n,i}) X_{n,j}`$

Slightly easier option : do it for the 2-class case. 

Hint: since there are only two options (say $`a`$ and $`b`$) and we know the second probability has to be $`Y_b = 1-Y_a`$, it's enough to worry about $`Y_a`$ alone, and this means we don't need two neurons computing two different $`\phi`$ values. Instead, just find one, $`Y_i = Prob(class=a)`$, and this can be implemented by a sigmoid (or logistic) non-linearity applied to $`\phi`$.

In this case the log likelihood is a sum over all items in the training set of this:

$` T_{n,i} \log Y_{n,i} + (1-T_{n,i}) \log (1- Y_{n,i})`$

which you can differentiate and reorganise to get the answer : the delta rule.

#### cross-entropy as appropriate loss function for classification

* Q : what is the "delta rule" appropriate for the weights going into softmax outputs then..? (A: the delta rule!)
* A : it's the same old delta rule - a striking coincidence? eg. do you see any connection between the sigmoid case and the softmax one?

#### back-propagation

We made a start.
See [back-propagation](../backprop)

<!--
* what is desired output for a classifier really? Excessive certainty is a problem.
 * ...we really want probabilities as outputs. Should be in [0,1] and sum to 1. We know this, so should "build it in" -> SOFTMAX reparameterization of y.
 * simplest possible parametric classification: make phi (the argument in softmax) linear in x.
:question: show that adding a constant to all the phi's has no effect on the y's. 

:question: show that if there are just two classes, softmax -> the "classic" sigmoid squashing function that appears in neural networks (at the output layer at least).

 * Exercises review
   * (does this indent?) So for classification tasks softmax is a natural thing to do
   * for example, we could compute a phi for each class (e.g. via a weighted sum of inputs)
   * you could think of that as one neuron per class, plus the softmax "post-processing" which makes bona fide probabilities as output values.
   * we're not talking about sampling from those probabilities mind you - just representing them: p(dog), p(cat), p(rat) 
   * for 2 classes things are special: instead of having 2 neurons (cat and dog) you might just as well have one, that uses a sigmoid non-linearity on a single phi.
   * in that case, we can interpret p(cat)=y and p(dog) = 1-y, so y=1 means certainty that it's a cat and y=0 means certainty it's a dog. 
 * Classifier: phi could be a weighted sum into a softmax (or sigmoid) - in retrospect, MF should have introduced this earlier on, to make all this "phi" talk more concrete.

 So much for the "M" in ML (Machine Learning), now for the "L"...
 
#### Loss functions
 * weight space and all that
 * data -> a loss surface (equiv. a "goodness" surface)
 * learning as search on this surface
 * nograds vs grads - if you can compute the gradients as well as the loss, that's potentially d "floats" of information about the surface compared to just 1 float (the evaluation alone). ie. _use gradients if you can!_
 * [Gradient descent](http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent)  ("SGD") in general. The "S" refers to use of minibatches, which is useful in practice for memory, and also seems useful for convergence (a noisy staggering seems to help)
 * e.g. regression: loss could be squared errors, -> "delta rule" learning, errror correction aspect of
 * but why squared...? We showed it's log likelihood, if Gaussian noise model. 
 * NB: it's possibly silly to use squared errors if you know something _isn't remotely_ Gaussian!
 * what about classification then? -> "cross entropy" loss function. [See also](http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/).
 * :question: show the latter leads to  -> delta rule again, somewhat remarkably!
   * I (Marcus) was rushed when explaining what the exercise was. Here is a [clarification](EXERCISE_Softmax_CrossEntropy.ipynb), except AARGH, gitlab does not render the LaTeX math in it (whereas github would, like [this](https://github.com/garibaldu/multicauseRBM/blob/master/Marcus/EXERCISE_Softmax_CrossEntropy.ipynb). Oh wait, instead of a notebook, I can just do [this](EX_softmax_crossentropy.md). 
   
#### simple "Perceptrons"
 * the most basic of parameterized models for the mapping X -> Y
 * weighted sum followed by sigmoid non-linearity (historically "Perceptron" refers to using a step non-linearity, going from 0 to 1 "instantly" at phi=0)
 * bias weight is good, acts as learnable threshold, can be treated as extra "zeroth" input that is always 1
 * perceptron's decision surface (where output changes) is a hyperplane, perpendicular to the weights vector
 * "learning" is movement of that hyperplane, in effect
 * Perceptron Learning Rule (often called the "delta rule") - guaranteed to find a separating hyperplane, if one exists
 * There's an [ipython notebook](https://gitlab.ecs.vuw.ac.nz/marcus/MachineLearningCOMP421/blob/master/notebooks/super-simple-Perceptron.ipynb) showing some of these aspects - download and run the last cell several times (from its random initial weight values) to see the effect of learning the weights on the hyperplane dividing the space.
 * XOR: a simple problem where separating hyperplane doesn't exist




## Lecture :three:

Here is a plan: everyone
 * [install anaconda, so that we can use jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html)
 * download and run [numpy-basics.ipynb](numpy-basic.ipynb) code - edit and play in the notebook
Oh and
 * make your first change to the repository for [this project](https://gitlab.ecs.vuw.ac.nz/marcus/MachineLearningCOMP421) - e.g. add an "issue" (question? answer?) or update this very page...
 
 -->
