#  exercise :two:   

Caveat: you don't _have_ to do this in python, but it's highly recommended as it is the dominant language for machine learning at the moment.

As well as python, we'll want to use the plotting package "matplotlib", and numerical array library "numpy", and a few other things. The easiest way by far is to get all these things in one shot, via [anaconda](https://www.anaconda.com/distribution/), so start by doing that. This should all work already on the ECS systems (I believe).

Note _Marsland_ has an introduction to all this in an appendix.

To run any of the "notebooks" (`XXX.ipynb` files) in what follows, go to the `notebooks` directory of the repo and invoke:

`>   jupyter notebook`

which should create a tab in a browser.
 * Work through [numpy-basics.ipynb](../notebooks/numpy-basics.ipynb), to get some familiarity with python / numpy and notebooks.
 * Then look through [super-simple-Perceptron.ipynb](../notebooks/super-simple-Perceptron.ipynb), which runs the "delta rule" in a single neuron with a logistic (sigmoid) function as its non-linearity. The data space is only 2 dimensional, which allows us to show the decision boundary visually.
 * Then have a look at the wonders of [autograd_example.ipynb](../notebooks/autograd_example.ipynb)
  
In super-simple-Perceptron, the cross-entropy loss $`\sum_n t_n \log y_n + (1-t_n) \log (1-y_n)`$ is given, *and* the resulting learning rule $`\Delta w_i = \eta (t-y) x_i`$ is also coded in. If you changed the loss function, you'd have to carefully change the learning rule too, or it wouldn't go "downhill" anymore. However, we now have autograd...

1. Change (or re-write from scratch) super-simple so that it uses `autograd` instead of an explicitly written gradient. Confirm it still works.
1. Change the loss to _something different_, and again confirm it still works (ie. confirm that the loss goes down over time. You might need to play with the learning rate a bit).
1. Change the transfer function to _something else_, and confirm it still works. 

Print out the results / notebook, and bring it to class on Tuesday, where we will mark them.

* Add to that page (hand-writing is fine) a note showing the following: a softmax with just 2 options (say, A and B) is equivalent to a sigmoid function outputting the probability of (say) A. Hints:
   * start with the softmax and argue towards the sigmoid
   * $`p(B) = 1-p(A)`$ if those are the only two classes
   * remember that we noticed you can add constant to softmax inputs without changing the output