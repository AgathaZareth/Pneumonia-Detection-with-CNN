# pneumonia-detection---convolutional-neural-network-CNN-

*This notebook is intended for educational purposes, all scerios are hypothetical, and any resulting model(s) should not be used for any medical purposes.*

---

## Introduction

According to the latest publication from Meticulous Research®, the global X-ray detectors market is expected to register a CAGR of 6% during the forecast period 2022–2029 to reach $4.30 billion by 2029. The growing adoption of digital X-ray detectors, rising demand for X-ray imaging in industrial and security markets, growing geriatric population coupled with rising prevalence of chronic diseases & respiratory infections, and increasing utilization of X-ray detectors for early diagnosis & clinical applications are considered to have a positive impact on the global X-ray detectors market.[[TOP 10 COMPANIES IN X-RAY DETECTORS MARKET]](https://meticulousblog.org/top-10-companies-in-x-ray-detectors-market/)

[GE Healthcare](https://www.gehealthcare.com/insights/article/achieving-greater-connectivity-in-radiology-through-digitization-and-ai) is among the top companies operating in the global digital radiography market.[[Digital Radiography Market Size to Reach USD 19.82 Billion in 2028, Says Reports and Data]](https://www.biospace.com/article/digital-radiography-market-size-to-reach-usd-19-82-billion-in-2028-says-reports-and-data/) GE Healthcare is also leading the way with integration of AI into their imagaing equipment and software.
 

## Business Understanding

They currently have on the market Critical Care Suite 2.01 (CCS), a collection of AI algorithms embedded on X-ray systems for automated measurements, case prioritization and quality control. The application automatically analyzes images on a GE X-ray system and highlights critical information on chest X-rays including Endotracheal Tube Positioning, Pneumothorax Triage and Notifications, and Quality Care Suite2 AI algorithms that operate in parallel and help technologists reduce image quality errors and improve efficiency. [[GE Healthcare]](https://apps.gehealthcare.com/app-products/critical-care-suite-2) 

GE Healthcare is expanding their collection of AI algorithms used when automatically analyzing images. One part of this expanded collection will screen for pneumonia and flag cases of concern. This will not only help GE Healthcare stay competative as the market grows but it will also increase the functionality of their CCS system. 

GE Healthcare needs a model that can successfully identify pneumonia. 


## Objectives

Build a model that can detect pneumonia to be integrate into GE Healthcare's collection of AI algorithms embedded in the Critical Care Suite 2.01 X-ray systems.


## Data Understanding 

The dataset comes from Kermany et al. on [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3). Images are from University of California San Diego, Guangzhou Women and Children's Medical Center and contains frontal chest X-rays of children and women. This is a large dataset with 5856 total radiographs; 1583 NORMAL and 4273 PNEUMONIA, both bacterial pneumonia and viral pneumonia. 


## Exploring the Data
<img src="/images/distribution.png" alt="distribution of classes" />


<img src="/images/first_20_full_data.png" alt="preview of first 20 images in the entire training set" />


---
---


# MODEL ARCHITECTURE

**OPTIMIZERS:**

SGD: "...recent studies show that Adam often leads to worse generalization performance than SGD for training deep neural networks on image classification tasks" [Adam vs. SGD: Closing the generalization gap on image classification](https://opt-ml.org/papers/2021/paper53.pdf).


**ACTIVATION FUNCTION:**

For hidden layers: ReLU, This is the standard activation function for hidden layers. There is no data to suggest deviating from this norm would be beneficial to this image classification task.

For output neron: Sigmoid, because this is a binary classification we can use sigmoid function to give probability of a given image being pneumonia. These probability predictions range from 0-1 making it easy to convert to percentages if so desired.  

**PADDING:**

For convolutional layers I will set padding equal to 'same'. This means there is one layer padding with blank pixels and the resulting pixels is the same size as the input image. It ensures that the filter is applied to all the elements of the input. Conversely, when padding is set to equal 'valid' there can be a loss of information, generally, elements on the right and the bottom of the image tend to be ignored. See below graphic from [this article](https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480):

padding='valid' is the first figure. The filter window stays inside the image.

padding='same' is the third figure. The output is the same size.

Visualization credits: [vdumoulin@GitHub](https://github.com/vdumoulin/conv_arithmetic)


<img src="https://i.stack.imgur.com/0rs9l.gif" alt="padding graphic" />



**LOSS:**

binary_crossentropy: This is a binary classification problem

**METRICS:**

accuracy: I am using accuracy because I want a reliable model that does a good job at accurately labeling cases. However, false negatives are a much bigger issue that false positives so in addition to accuracy I will be focusing on pneumonia recall score from the classification report as a secondary metric for guaging model performance. And more specifically, I will be focusing on the difference between training recall scores and testing recall scores, they should be as similar as possible to show that the final model will reliably perform at the same level on new unseen data/cases as it did with the training data. 

**MAXIMUM POOLING** 

Calculate the maximum value for each patch of the feature map. The result of using a pooling layer and creating down sampled or pooled feature maps is a summarized version of the features detected in the input. [A Gentle Introduction to Pooling Layers for Convolutional Neural Networks](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

**L2 KERNAL REGULARIZATION**

l2 regularization uses a lambda coefficient of .005.

Performing L2 regularization encourages the weight values towards zero (but not exactly zero). Smaller weights reduce the impact of the hidden neurons. In that case, those hidden neurons become neglectable and the overall complexity of the neural network gets reduced, less complex models typically avoid modeling noise in the data, and therefore, there is no overfitting. 

Choosing the right lambda coeffient value:
- If value is too high you run the risk of underfitting your data. Your model won’t learn enough about the training data to make useful predictions.
- If value is too low you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won’t be able to generalize to new data.

**DROPOUT**

During dropout, some neurons get deactivated with a random probability P to reduce model complexity resulting in less overfitting. I am using P=.03

---
---

# Models

## `base_model`
Baseline fully connected model without convolutional layers:

### `base_model.summary()`

<img src="/images/base_summary.png" alt="base_model.summary()" />

### Loss and Accuracy across epochs

<img src="/images/base_la_plot.png" alt="baseline loss and accuacy across epochs plots" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/base_train_cm_cr.png" alt="baseline training data confusion matrix and classification report" />

**Test**

<img src="/images/base_test_cm_cr.png" alt="baseline test data confusion matrix and classification report" />

## `base_cnn`
Add convolutional layers to `base_model`

### `base_cnn.summary()`

<img src="/images/bass_cnn_summary.png" alt="base_cnn.summary()" />

### Loss and Accuracy across epochs

<img src="/images/base_cnn_la_plot.png" alt="baseline cnn models loss and accuacy across epochs plots" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/base_cnn_train_cm_cr.png" alt="baseline cnn training data confusion matrix and classification report" />

**Test**

<img src="/images/base_cnn_test_cm_cr.png" alt="baseline cnn test data confusion matrix and classification report" />

## `reg_cnn`
Add l2 kernel regularization to `base_cnn` model

### `reg_cnn.summary()`

<img src="/images/name.png" alt="name.summary()" />

### Loss and Accuracy across epochs

<img src="/images/name.png" alt="name loss and accuacy across epochs plots" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/name.png" alt="name training data confusion matrix and classification report" />

**Test**

<img src="/images/name.png" alt="name test data confusion matrix and classification report" />

## `reduced_nodes`
Reduce number of nodes in each layer by half

### `reduced_nodes.summary()`

<img src="/images/name.png" alt="text" />

### Loss and Accuracy across epochs

<img src="/images/name.png" alt="text" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/name.png" alt="text" />

**Test**

<img src="/images/name.png" alt="text" />

## `dropout`
Add dropout to `reduced_nodes` model

### `dropout.summary()`

<img src="/images/name.png" alt="text" />

### Loss and Accuracy across epochs

<img src="/images/name.png" alt="text" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/name.png" alt="text" />

**Test**

<img src="/images/name.png" alt="text" />



<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />
<img src="/images/name.png" alt="text" />









