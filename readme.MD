# Gated Shape CNNs
Implementation of [this paper](https://arxiv.org/abs/1907.05740) for semantic segmentation. Written using tensorflow 2.0, and the model is implemented as a subclassed `tf.keras.Model`. 

There are some differences between this implementation and the paper: 
- Use normal resnet instead of wide resnet
- Use generalised dice loss instead of cross entropy for the edge segmentation
- No synchronised batch norm

# IOU
At the moment the best model I have trained is much worse than the paper, but this was because I did not replace the final downsampling layers in the resnet with atrous convolutions. I have done this now, but need to get access to a bunch of gpus to train on a decent batch size.
Implementation| road| sidewalk | building | wall | fence | pole | traffic light | traffic sign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle
| ---         | --- | ---      | ---      | --- | ---   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| Paper       | 98.3|86.3      |93.3      |55.8 |64     |70.8|75.9|83.1|93|65.1|95.2|85.3|67.9|96|80.8|91.2|83.3|69.6|80.4|
| This repo | 95.4|72.3|86.5|49.1|49|40.6|40.7|56.9|87.7|53.4|89|70.5|52.8|90.1|71.9|76.5|58.7|50|63.8

# Todo 
- Make synchornised batch norm and train on larger batch

