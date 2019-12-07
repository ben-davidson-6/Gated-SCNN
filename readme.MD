# Gated Shape CNNs
```diff
- work in progress
```
Implementation of [this paper](https://arxiv.org/abs/1907.05740) for semantic segmentation. Written using tensorflow 2.0, and the model is implemented as a subclassed `tf.keras.Model`. 

There are some differences between this implementation and the paper: 
- Inception instead of resnet backbone
- Use generalised dice loss instead of cross entropy

# todo
- understand tf 2.0 better as I think the custom training is not using prefetching 
- get a gpu to train on something non-trivial and provide the weights
