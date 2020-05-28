# 1.0.2
### 28.05.2020
- forced the network to always use batch size > 1 if training, because of a batch normalisation bug in tensorflow which would result in nans for batch size = 1

# 1.0.1 Initial version!
### 26.05.2020
- added tests
- fixed bugs if there were no gradients

# 1.0.0 Initial version!
### 22.05.2020