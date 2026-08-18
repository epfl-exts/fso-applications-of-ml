<img src="static/EXTS_Logo.png" width="125px" align="right">

# Applications in Machine Learning: Image Classification

This repository provides the resources for the session and accompanying hands-on exercises on **Image Classification** at the **EPFL Extension School Workshop - Applications in Machine Learning**.

**Dataset**

We use a set of 3'624 images of eight animal classes — brown bear, polar bear, giant panda, red panda, lion, tiger, racoon and red fox — collected from openly licensed sources. The images live in `data/`, one folder per class.

**Image Classification**

Images are just numbers: a colour image is a grid of pixels with three values each. This hands-on exercise builds up three progressively better ways of turning those numbers into features a classifier can use:

1. **Raw pixels** — downsample each image to 32 x 32 and use the pixel values directly.
2. **RGB colour profiles** — summarise each image by the histogram of its red, green and blue channels.
3. **Neural network features** — pass each image through MobileNetV2, a convolutional network pre-trained on ImageNet, and use the 1'280-dimensional vector it produces.

In all three cases the *classifier* is the same simple `RidgeClassifier`, so the comparison isolates the effect of the features. The pre-trained network is used only as a feature extractor — nothing is trained on the images themselves. This is called **transfer learning**.

The neural network features are pre-computed and stored in `data_nn.npz`, so the exercise runs without waiting for the network. To recompute them from the images, call `extract_neural_network_features(rerun=True)`.

**Getting started:**

Start with loading settings and functions. If you want to execute a cell, make sure it is selected and then press `SHIFT`+`ENTER` or the `'Play'` button.

The notebook expects the kernel's working directory to be this folder — all paths (`data/`, `data_nn.npz`, `./scripts/utils.py`) are relative to it.
