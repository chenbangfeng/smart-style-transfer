# Modified implementation of 'A Neural Algorithm of Artistic Style'

- [Torch7](https://github.com/torch/torch7)
- [gSLICr-torch](https://github.com/jhjin/gSLICr-torch) (used for segmentation)
- CUDA 6.5+ (unless running on CPU -- see below)

## Usage

First, download the models by running the download script:

```
bash download_models.sh
```

This downloads the model weights for the VGG and Inception networks.

Basic usage:

```
qlua smart_style_transfer.lua -style_image <style.jpg> -content_image <content.jpg> -mask_labels <segment.dat>
```

where `style.jpg` is the image that provides the style of the final generated image, and `content.jpg` is the image that provides the content. `segment.dat` is a segmentation of the content image used to product better style weights.

This generates an image using the VGG-19 network by Karen Simonyan and Andrew Zisserman (http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

## Acknowledgements

Thanks to the [Bethge Group](http://bethgelab.org/deepneuralart/) for providing the weights to the normalized VGG network used here.
