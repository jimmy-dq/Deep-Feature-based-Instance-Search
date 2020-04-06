# Deep-Feature-based-Instance-Search

Contact: [Jimmy.Wu@my.cityu.edu.hk](mailto:Jimmy.Wu@my.cityu.edu.hk). Any questions or discussion are welcome! 

## Introduction
This project presents the two instance search methods with pre-trained deep features extracted from deep neural networks, e.g., ResNet-18 and VGG-11.

## Features
For the methodology-1, we use the deep features extracted from the global averaging pooling layer of ResNet-18 for instance search. The obtained feature vectors are 512-D, which is not very high-dimensional and is suitable for retrieval. For the methodology-2, we employ VGGNet (i.e., VGG-11) as our feature extractor. Note that VGGNet contains two modules: 1) feature module; 2) classification module. In this project, we use the feature module that consists multiple convolutional layers to extract features.
