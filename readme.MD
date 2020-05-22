# Gated-Shape CNN for Semantic Segmentation in Tensorflow 2.0
Implementation of [Gated-Shape CNN for Semantic Segmentation (ICCV 2019)](https://arxiv.org/abs/1907.05740) for semantic segmentation. Written using tensorflow 2.2.0rc3, and the model is implemented as a subclassed `tf.keras.Model`. 

## Performance on CityScapes

Implementation| mean     | road | sidewalk  | building | wall | fence | pole| traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle|
| ---         |    ---   | ---  | ---       | ---      | ---  | ---   | --- | ---           | ---          | ---        | ---     | --- | ---    | ---   | --- | ---   | --- | ---   | ---        | ---    |
| Paper       |   80.8   | 98.3 | 86.3      |93.3      |55.8  |64     |70.8 |75.9           |83.1          |93          |65.1     |95.2 |85.3    |67.9   |96   |80.8   |91.2 |83.3   |69.6        |80.4    |
| This repo   |   77.7   | 97.8 | 83.2      |92.4      |55.8  |59.8   |64.4 |67.6           |77.5          |92.4        |63       |94.9 |81.9    |63     |95   |80.8   |86   |78.3   |65.2        |77.1   |

[Model weights are here](https://drive.google.com/open?id=1wZRoMeCP25Qcrm33aMjT1Hm6zynS4dsB)

## Some examples
<div align="center">
<img src="images/out.gif" alt="stuttgart_02" style="display: block; margin-left: auto; margin-right: auto; width: 50%;" border="10" />
</div>

## Install
Versions of the code are tracked with tags, if you want to install version 1.0.0 then do:

```pip install git+https://github.com/ben-davidson-6/Gated-SCNN.git@1.0.0```

note that you need `tensorflow==2.2.0`

## Training on your own data
### Just give me the network!
```python
import gated_shape_cnn.model
number_classes = 10
# will create a tf.keras.model.Model
model = gated_shape_cnn.model.GSCNN(n_classes=number_classes)
```
### Using the full training loop
You have two options

### Inheriting from `gated_shape_cnn.training.Dataset`
To use the existing dataset class you need to have all of your images, labels, and edge boundaries prepared ahead of time, and in the following format.

1. First make sure you have the data in the right format
    - Images are 3 channels `[h, w, 3]` you should be able to use anything that works inside of `tf.io.decode_image`. Note this does not 
    - Your segmentation should be flat, so of the shape `[h, w, 1]` with the last channel containing the class id and masks should be .png's
    - Edge segmentations should follow the same format as segmentations. If you do not have edge segmentations you can create them with `gated_shape_cnn.training.utils.flat_label_to_edge_label` 
2. Implement a method in your class which inherits from `gated_shape_cnn.training.Dataset`.
    - `get_paths` should return 3 lists which contain the paths to images, paths to semantic segs, and paths to edges such that the ith data in each list relates to the ith image
    ```python
    import gated_shape_cnn.datasets.cityscapes
    import gated_shape_cnn.datasets.cityscapes.raw_dataset
    from gated_shape_cnn.training import Dataset
    
    class CityScapes(Dataset):
    
        def __init__(
                self,
                batch_size,
                network_input_h,
                network_input_w,
                max_crop_downsample,
                colour_aug_factor,
                debug,
                data_dir):
            super(CityScapes, self).__init__(
               gated_shape_cnn.datasets.cityscapes.N_CLASSES,
                batch_size,
                network_input_h,
                network_input_w,
                max_crop_downsample,
                colour_aug_factor,
                debug)
            self.raw_data = gated_shape_cnn.datasets.cityscapes.raw_dataset.CityScapesRaw(data_dir)
    
        def get_paths(self, train):
            """
            :param train: 
            :return image_paths, label_paths, edge_paths:
                image_path[0] -> path to image 0 
                label_paths[0] -> path to semantic seg of image 0
                edge_paths[0] -> path to edge seg of label 0
            """
            split = gated_shape_cnn.datasets.cityscapes.TRAIN if train else gated_shape_cnn.datasets.cityscapes.VAL
            paths = self.raw_data.dataset_paths(split)
            image_paths, label_paths, edge_paths = zip(*paths)
            return list(image_paths), list(label_paths), list(edge_paths)
    ```
3. train your model using the `gated_shape_cnn.training.train_model`

    ```python
    from gated_shape_cnn.training import train_model
   
    train_model(
        n_classes=instance_of_subclassed.n_classes,
        train_data=instance_of_subclassed.build_training_dataset(),
        val_data=instance_of_subclassed.build_validation_dataset(),
        optimiser=optimiser,
        epochs=300,
        log_dir='./logs',
        model_dir='./logs/model',
        accum_iterations=4,
        loss_weights=(1., 20., 1., 1.))
    ```
## Building your own dataset
You do not have to go through the `Dataset` class to use the training loop, all you need to provide are two `tf.data.Dataset`'s which when iterated over are of the form
```python
for im, label, edge_label in dataset:
    # im         [b, h, w, 3]       tf.float32 note this is not normalised, as the xception preproccesing is part of the model
    # label      [b, h, w, classes] tf.float32
    # edge_label [b, h, w, 2]       tf.float32
    pass
```
You can then feed these into `train_model` for `train_data` and `val_data`.

## Differences to paper
- Use Xception instead of WideResnet
- Only replace the final downsampling layers with atrous convolution (usually you replace both)
- Use generalised dice loss instead of cross entropy for the edge segmentation
- I accumulate gradients over iterations as I do not have 8 GPUS! Instead of batch size of 2 synchronised on 8 GPUS I have a batch of size 4 accumulating 4 passes
- train on a smaller resolution 700x700 versus 800x800

## Todo 

- [ ] Write tests
- [ ] add ci/cd so look like I know what I am doing
- [ ] build version using keras.fit 

