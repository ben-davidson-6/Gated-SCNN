# Gated Shape CNNs
Implementation of [this paper](https://arxiv.org/abs/1907.05740) for semantic segmentation. Written using tensorflow 2.0, and the model is implemented as a subclassed `tf.keras.Model`. 

There are some differences between this implementation and the paper: 
- Inception instead of resnet backbone
- Use generalised dice loss instead of cross entropy for the edge segmentation


# todo
- replicate similar performance on cityscapes
