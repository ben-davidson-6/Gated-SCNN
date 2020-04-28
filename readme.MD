# Gated Shape CNNs
Implementation of [this paper](https://arxiv.org/abs/1907.05740) for semantic segmentation. Written using tensorflow 2.2.0rc, and the model is implemented as a subclassed `tf.keras.Model`. 

There are some differences between this implementation and the paper: 
- Use Xception instead of WideResnet
- Use generalised dice loss instead of cross entropy for the edge segmentation

# Requires
Only tested on tensorflow==2.2.0rc3. There are definitely things which will not work with earlier versions of tensorflow.

# Model
Xception with some downsampling convolutions replaced with atrous convolutions.

# IOU
Implementation| mean  | road | sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
| ---         |    ---   | --- | ---       | ---      | --- | ---   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| Paper       |   80.8    | 98.3|86.3       |93.3      |55.8 |64     |70.8|75.9|83.1|93|65.1|95.2|85.3|67.9|96|80.8|91.2|83.3|69.6|80.4|
| This repo   |   66    | 95.4|72.3       |86.5|49.1|49|40.6|40.7|56.9|87.7|53.4|89|70.5|52.8|90.1|71.9|76.5|58.7|50|63.8

The reason for reduced performance, I think, is I have trained on a smaller resolution (513x513 instead of 800x800) and I have used a smaller batch size (8 instead of 16) as I cannot get access to multiple GPUs.
# Todo 
- Train better model on multiple gpus
