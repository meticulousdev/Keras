[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

$\mathcal{H}(x)$: Underlying mapping to be fit by a few stacked layers (not necerssarily the entire net) 

If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, i.e., $\mathcal{H}(x)-x$ (assuming that the input and output are of the same dimensions). So rather than expect stacked layers to approximate H(x), we explicitly let these layers approximate a residual function $\mathcal{F}(x) := \mathcal{H}(x) - x$.
...(중략)
With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

Source: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

$y_l = h(x_l) + \mathcal{F}(x_l, W_l)$

$x_{l+1} = f(y_l)$


$x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$

$x_{l+2} = x_{l+1} + \mathcal{F}(x_{l+1}, W_{l+1}) = x_l + \mathcal{F}(x_l,W_l ) + \mathcal{F}(x_{l+1}, W_{l+1})$

$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$
