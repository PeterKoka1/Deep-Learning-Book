# Deep-Learning-Book
Some math operations and applied examples from Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. The online textbook can be found at http://www.deeplearningbook.org/.

### Kullback-Leibler-divergence-tests.ipynb
Computing the DKL convergence from scratch with applications to skewed and multi-tailed distributions. 
Gradients also displayed

### Activation-Saturation.py
Showing how improper weight initialization causes activations to saturate during forward and backward passes along the gradient. Using the modified Xavier weight matrix initialization to solve saturation. Huge props to all material and notes and an awesome course @CS231N from Stanford university (http://cs231n.stanford.edu/index.html).

### KNN-cat.py
Finding minimum L1 norm to classify the image of a cat (naive KNN with K=1) from scratch.
Train set was 5 images: a cat, person, fire hydrant, book, and computer. Test image was another picture of a cat. All images scraped from the web

### Learning_XOR.ipynb
Learning the XOR ("exclusive or") function with a feedforward network from scratch

### Multi-Class-SVM_cat.py
Unvectorized approach to multi-class Support Vector Machine. Computing L2 loss from first iteration of gradient. Weights all set to 0.1. 
Goal: image recognition with 5 classes.

### Normal_Equation_on_Time_Series.ipynb
Normal equation on equity price (GOOGL and AMZN) time series. Using L2 norm to display gradient
