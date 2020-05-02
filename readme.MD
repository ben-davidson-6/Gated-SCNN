# Gated Shape CNNs
Implementation of [this paper](https://arxiv.org/abs/1907.05740) for semantic segmentation. Written using tensorflow 2.2.0rc3, and the model is implemented as a subclassed `tf.keras.Model`. 

There are some differences between this implementation and the paper: 
- Use Xception instead of WideResnet
- Only replace the final downsampling layers with atrous convolution (usually you replace both)
- Use generalised dice loss instead of cross entropy for the edge segmentation
- I accumulate gradients over iterations as I do not have 8 GPUS!
- train on a much smaller resolution 5123x513 versus 800x800

# IOU on CityScapes

Implementation| mean     | road | sidewalk  | building | wall | fence | pole| traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle|
| ---         |    ---   | ---  | ---       | ---      | ---  | ---   | --- | ---           | ---          | ---        | ---     | --- | ---    | ---   | --- | ---   | --- | ---   | ---        | ---    |
| Paper       |   80.8   | 98.3 | 86.3      |93.3      |55.8  |64     |70.8 |75.9           |83.1          |93          |65.1     |95.2 |85.3    |67.9   |96   |80.8   |91.2 |83.3   |69.6        |80.4    |
| This repo   |   74.3   | 97.7 | 82.6      |91.2      |48.7  |53.1   |62.2 |64.7           |75.1          |91.9        |62.4     |93.5 |80.5    |60     |94.1 |71.8   |80.5 |66.6   |60.3        |75.3    |

<img src="images/out.gif" alt="stuttgart_02" style="display: block; margin-left: auto; margin-right: auto; width: 50%;" border="10" />
# Requires
Only tested on tensorflow==2.2.0rc3. There are definitely things which will not work with earlier versions of tensorflow.

# Model
The backbone network is `tf.keras.application.Xception` with the final downsampling convolution and max pooling layers replaced with the atrous convolution and the identity respectively. Starts from the imagenet weights tensorflow provides.



# Training on your own data
If you want to train on your own data then you need to 
- subclass the Dataset class and define two methods for an example see `datasets.cityscapes.dataset`:
- alternatively you just need to pass a training and validation `tf.data.Dataset` to the a `gscnn.train_and_evaluate.Trainer` which you   can iterate over like:
- input needs to be have 3 channels (RGB) but would be easy to make more general, but you'll have to write some code
```python
for im, label, edge_label in self.train_dataset:
    # im         [b, h, w, 3]       tf.float32
    # label      [b, h, w, classes] tf.float32
    # edge_label [b, h, w, 2]       tf.float32
```
- Define training setup, see `cityscapes_model.py`

# Todo 
- Write tests
- add ci/cd so look like I know what I am doing
- retrain with better boundaries as I made a mistake in the model mentioned here, which meant I had some extra boundaries, for example the car shouldnt generate a boundary but I did when building the training data
- try to get better performance

