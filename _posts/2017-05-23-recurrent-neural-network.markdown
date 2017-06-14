---
layout: post
title:  "Recurrent Neural Network"
date:   2017-05-23 15:15:28 +0200
categories: main
---
<h1>Basic</h1>
In contrast with densely connected neural network or [Convolution Neural Network][cnn-link], the recurrent neural network delivers a big variety in deep learning model. Let's have a essential picture of this model at first. RNN is a kind of <strong>State-Space Model</strong> that has the following structure
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/state-space-model.png" width="65%">
  <div class="figcaption">a state space model</div>
</div>

The left part of this diagram is the system model, where $$a_n$$ is a vectorial function, $$x_n$$ is the current value of state, $$x_{n+1}$$ denotes the value of subsequent state, $$z^{-1 y}$$ is a time delay unit and $$s_n$$ is the dynamic noise. $$x_{n+1}$$ has the following definition form:
 \begin{equation}
 x_{n+1} = a_n(x_n, s_n)
 \end{equation} The right part is the measurement model in this state space system, where $$v_n$$ the noise in measure  and $$y_n$$ the observable output that defined as 
 \begin{equation}
 y_n = b_n(x_n, v_n)
 \end{equation}
 The evolution process varies over time is depicted as follows:
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/state-space-model-evolution.png" width="75%">
  <div class="figcaption">Evolution of state space model over time steps</div>
</div>

As we can see that the structure of this model takes the time steps into account. Actually this is the [Bayesian Filtering system][bayes-link].  Our RNN has also the similar structure, whose evolution structure over time depicted as given by:
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/rnn.png" width="65%">
</div>

This structure depicts a interesting characters: the generated hidden state in previous state $$h_{t-1}$$ is fed in the subsequent state $$h_t$$. The weight connecting the input and hidden state $$W_{xh}$$ are reused in each time step $$t$$, at each time step there will be a output $$y_t$$ generated. Formally this can be defined as follows:

$$\begin{align}
\begin{split}
y_t &= W_{hy} h_t \\
&= W_{hy}  f_w(h_{t-1}, x_t)\\
&= W_{hy} \tanh(W_{hh} h_{t-1} + W_{xh}x_t)
\end{split}
\end{align}$$

When this model works over time steps, the context information (or prior information) in hidden neurons will be stored and passed in the next time step so that learning task that extends over a time period can be performed.

<h1>Image Captioning</h1>
As a brief introduction of the RNN's structure was given. Here is an example that uses this model to caption a image. Image captioning means, say, you give a image to a model. This model learns the saliencies firstly. After running our model with this saliencies you'll get a consice description about what's going on in this given image.  For image captioning we need extra information that represents the image to be captioned. This information can be learned by other learning model, i.e. convolution neural network. The captioning vocabulary consists of string library, but for simplifying the learning process this vocabulary is converted in to integers that can be easily handled as vectors. At the test time we get a bunch of integers as return results, then we decode it into the original word string  correspondingly. 

<h2>Training Stage</h2>
At training time features of images are trained at first by convolution network. These features are projected with this $$p(f,w)$$ affine function to initial hidden state of our network. The vocabulary indexes are coded in a word embedding layer that maps the word to matrix that we can perform our for- and -backward propagation numerically. The training caption data  consists of a image looks like as follows:
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/rnn_train.png" width="80%">
</div>
<h2>Training Stage</h2>

Each caption is tagged with some extra information likes $$<START>$$ and $$<END>$$ tokens. All the captions vocabularies are handled as matrix, cause we can't use them directly, say for instance,  in forward propagation flow.
During the training the weight of corresponding words will be updated, which also reflects the distribution of vocabularies. Also at each time step the last hidden state is used to forward propagate the "context information" over time steps. This context information implies the potential connections in original training data. The word score corresponding to features are calculated in a affine layer and fed into softmax layer to calculate the loss function. The yellow arrow indicates the direction of back propagation. The whole training task is actually not so complicated. The normal [update rules][opt-link] can also be used for updating the parameters here. The back progapation works symetrically as we do in normal case.



<h2>Test Stage</h2>

The test process is shown in the figure below. The feature conversion is just same as we did in training process. Now the input to our model is little different. At the beginning of test the first word feed to model is $$<START>$$ token.  The feature will be used in and combined with the hidden state. The rest stays same as in training state excpet the last step, at which we will get a word that has the maximal score. The word with maximal score is stored in captions vectors and used in subsequent iterations. After iterations the caption vectors will be returned and decoded in corresponding text. The procedure repeats over all the features. The output is just then a couple of words that we got in each time step.



<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/rnn_test.png" width="80%">
</div>

<h1>Pros vs. Cons</h1>
The workflow of this model in the area of image caption has been briefly presented. The hidden state are always newly generated as training. So as the error propagates backwards, this should be summed in total time steps but initial hidden state. The content information in previous neurons has been stored and forwards propagated via hidden state and the relation between features and words over vocabulary is mapped in the the affine layer and the corresponding importance is reflected in the vocabulary weights in this layer. Here the vocabulary is also important, or in other words the distribution of words in vocabulary is very essential to caption images. If the words are not good selected enough, the caption will be meanless of human. 

 
The structure of this model is quite simple to understand. Each hidden state just stacked over time steps that lurks the potential problem. Let's recap our training process again. We feed our features as inputs to RNN, and combined the non-linearity, infinitesimal changes to update our parameters over time distance. This may do not effect the training or be neglectable. Or if we have a big change in inputs that are unfortunately not measured by the gradient as time changes. This so-called [Vanishing-gradients][vanish-link]  problem downgrades the learning that has long-term dependencies in gradient-based models.For solving this problem we can use non-linear sequential sequence state model like [Long Short Term Memory]({{ site.baseurl }}{% post_url 2017-06-02-long-short-term-memory  %})


















[cnn-link]:https://keeperovswords.github.io/DeepConvolutionNetwork/main/2017/04/11/deep-learning.html
[bayes-link]:https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation
[opt-link]:https://keeperovswords.github.io/Optimization/
[vanish-link]:https://en.wikipedia.org/wiki/Vanishing_gradient_problem
