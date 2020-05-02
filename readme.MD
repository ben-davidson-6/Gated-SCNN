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


Implementation| mean     | road | sidewalk  | building | wall | fence | pole| traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle|
| ---         |    ---   | ---  | ---       | ---      | ---  | ---   | --- | ---           | ---          | ---        | ---     | --- | ---    | ---   | --- | ---   | --- | ---   | ---        | ---    |
| Paper       |   80.8   | 98.3 | 86.3      |93.3      |55.8  |64     |70.8 |75.9           |83.1          |93          |65.1     |95.2 |85.3    |67.9   |96   |80.8   |91.2 |83.3   |69.6        |80.4    |
| This repo   |   74.3   | 97.7 | 82.6      |91.2      |48.7  |53.1   |62.2 |64.7           |75.1          |91.9        |62.4     |93.5 |80.5    |60     |94.1 |71.8   |80.5 |66.6   |60.3        |75.3    |

The reason for reduced performance, I think, is I have trained on a smaller resolution (513x513 instead of 800x800) and I have used a smaller batch size (8 instead of 16) as I cannot get access to multiple GPUs. I also only replace the final downsampling convolution with an atrous conv, rather than the last two downsampling.
# Todo 
- Train better model on multiple gpus
