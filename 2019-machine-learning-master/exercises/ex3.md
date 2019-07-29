#  exercise :three:   

Caveat: you don't _have_ to do this in python, but it's highly recommended as it is the dominant language for machine learning at the moment.

The goal here is to go from having a working perceptron to a working neural net, that you wrote, and which makes use of autograd. 

Frameworks such as torch / pytorch and keras / tensorflow (and others) are of course fantastic for doing things at scale, and for making use of CUDA/GPU - and you should feel free to jump to those if you want to, but _after_ this exercise.

* You should be able to easily mess with the number of hidden layers, 
* and have different non-linearities. (but don't worry about richer architectures such as "skip" connections), including ReLUs and softmax
* be able to read in a data set in some standard format - I'd look to scikit-learn (python module) to do this: keep it simple! Pick a modest sized classification problem. I recommend .... (digits in scikit-learn)
* use autograd to get the gradients
* start from random initial weights and use "vanilla" (no fancy optimizers) to train your net on that data set
* plot the loss-vs-epochs, from 3 random restarts, all on the same graph. (NB: 1 epoch is just once through the data set). "matplotlib" can easily make the plots. You can use the whole training set if you like (ie. slicing it into randomized minibatches and thus doing SGD isn't necessary).
* if you get that going, try adding some momentum, and plot some the new loss-vs-epochs lines.

Print out the results / notebook, and bring it to class on Tuesday, where we will mark them.

I would also like to collect these ones in - I'm keen to see how you coded it! There's no "one way". Just don't simply grab someone else's from the web :) as you won't learn as much.
