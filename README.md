
# Keras Semantic Segmentation
___

This is an enhancement of the Semantic Segmentation using Keras, with implementation of Free Convolutional Networks (FCNs).

**Recent Updates:**

- 2019-02-22: Implemented common FCNs and added support for Geo-tiff Images, a great feature for remote sensing images.
- 2019-03-07: Successfully tested on VOC2012 and Inria datasets.

**Coming Next:**

- More State-of-the-art FCN architectures.
- Better support for different output strides and diverse open datasets like VOC, CityScapes, ADE20K, MSCOCO.
- Improve flexibility in data formats.

**Currently Available Backbones and FCNs:**

- ResNet_v2 variants
- VGG configurations
- Xception-41
- Various FCNs including U-net, SegNet, PSPNet, RefineNet, Deeplab v3, Deeplab v3+, and Dense ASPP

**To Be Implemented:**

- DenseNet backbone
- ICNet FCN

### Folder Structure and Environment
This project assumes you have Python 3.6 installed along with libraries such as `tensorflow-gpu`, `Keras`, `opencv`, `PIL`, `numpy`, `matplotlib`, `tqdm`, `GDAL`, `scikit-learn`.

Detailed folder structure and guidelines for dataset generation and conversion, model training, prediction, and evaluation are provided in the main README.md

### Dataset
Testing has been done on datasets such as WHU Building, Inria Aerial Building Labeling, ISPRS 2D Semantic Labeling Benchmark, Massachusetts Roads and Buildings, VOC2012, CityScapes, ADE20K.

Please reach out to the current project maintainer cinderellab for any questions or suggestions.