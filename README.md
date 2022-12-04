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

<img src="/images/reg_cnn_summary.png" alt="reg_cnn.summary()" />

### Loss and Accuracy across epochs

<img src="/images/reg_cnn_la_plot.png" alt="reg_cnn loss and accuacy across epochs plots" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/reg_cnn_train_cm_cr.png" alt="reg_cnn training data confusion matrix and classification report" />

**Test**

<img src="/images/reg_cnn_test_cm_cr.png" alt="reg_cnn test data confusion matrix and classification report" />

## `reduced_nodes`
Reduce number of nodes in each layer by half

### `reduced_nodes.summary()`

<img src="/images/reduced_nodes_summary.png" alt="reduced_nodes.summary()" />

### Loss and Accuracy across epochs

<img src="/images/reduced_nodes_la_plot.png" alt="reduced_nodes loss and accuacy across epochs plots" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/reduced_nodes_train_cm_cr.png" alt="reduced_nodes training data confusion matrix and classification report" />

**Test**

<img src="/images/reduced_nodes_test_cm_cr.png" alt="reduced_nodes test data confusion matrix and classification report" />

## `dropout`
Add dropout to `reduced_nodes` model

### `dropout.summary()`

<img src="/images/dropout_summary.png" alt="dropout.summary()" />

### Loss and Accuracy across epochs

<img src="/images/dropout_la_plot.png" alt="dropout loss and accuacy across epochs plots" />

### Confusion Matrices and Classification Reports

**Train**

<img src="/images/dropout_train_cm_cr.png" alt="dropout training data confusion matrix and classification report" />

**Test**

<img src="/images/dropout_test_cm_cr.png" alt="dropout test data confusion matrix and classification report" />


# Compare the models

`compare` df:

<img src="/images/compare.png" alt="train and test data for each model, accuracy and pneumonia recall scores" />

`metrics_df`:

<img src="/images/metrics_df.png" alt="each models train loss, test loss, train accuracy, test accuracy, and the differences between both" />

My goal for adding dropout was to close the gap between the training and test loss difference and train and test accuracy difference. The `metrics_df` shows that this `dropout` model did infact reduce that gap from the previous model iteration. However, the `compare` df shows that while the accuracy improved with this `dropout` model, the pneumonia recall score for test data dropped. You can also explore this in greater detail by comparing the confusion matrices and classification reports of the train data from the last two models. 

At this point I need to move forward with selecting a final model. I can chose the higher accuracy `dropout` model, or the higher recall score of `reduced_nodes` model.

Because the intent of this model is to flag cases of concern, I will go with the higher recall score model, `reduced_nodes`, so that fewer cases are given a false negative label of NORMAL. This may result in increased number of cases prioritized for immediate review from radiologists however, I will capture as many true pneumonia cases as possible which in the end will result in expidited treatment for these patients. 

# Final Model
For the above mentioned reasons I have chosen my final model to be the `reduced_nodes` model. For the `final_model` I downsampled the pneumonia class to match that of the normal class. This will bring the total number of training images to 2698 with an even split of both classes. I then selected 25 from each class for validation data. 

## Architecture

<img src="/images/final_model_arch.png" alt="final_model architecture" />


## Summary

<img src="/images/final_model_summary.png" alt="final_model.summary()" />

## Loss and Accuracy across epochs

### Plot

<img src="/images/final_model_la_plot.png" alt="final_model loss and accuacy across epochs plots" />

**NOTES:** The early stopping was triggered after 42 epochs. It took approximately 180 seconds per epoch on this machine. It is clear from the above plots that this model is still slightly over trained (take note of the scale of y axis). Next Steps will discuss options to address this.

### Compare with `compare` df

<img src="/images/compare_with_final.png" alt="each models accuracy and pneumonia recall scores for train and test data" />

**NOTES:** There is still a very high accuracy with both train and test data, ~98.7% and ~91.2% respectively. Additionally, the differences between training and test, loss and accuracy are the lowest in this final model. Remember, this is the `reduce_nodes` model and the only difference is the amount of data that has been used to train the model. 

## Confusion Matrices and Classification Reports

**Train**

<img src="/images/final_train_cm_cr.png" alt="final_model training data confusion matrix and classification report" />

**Test**

<img src="/images/final_test_cm_cr.png" alt="final_model test data confusion matrix and classification report" />

## View misclassified images from testing data

<img src="/images/final_model_misclassified.png" alt="Missclassified images from final models test data" />

## Final evaluation

### `metrics_df` 

<img src="/images/metrics_df_with_final.png" alt="each models train loss, test loss, train accuracy, test accuracy, and the differences between both" />

**NOTES:** From the `compare` df you can see the testing data accuracy scored highest with this model. You can also see in both training and test data pneumonia recall score is the same. This is important because we want to avoid false negatives slipping by without immediate attention called to them. While it is not the highest of all the models it is important they match because we want a reliable model that is outputing similar results on unseen data  as it does on testing data. As noted above, from the `metrics_df` you can see there is still a very high accuracy with both train and test data, ~98.7% and ~91.2% respectively. Additionally, the differences between training and test, loss and accuracy are the lowest in this final model. Remember, this is the `reduce_nodes` model and the only difference is the amount of data that has been used to train the model. This demonstartes the potential for this model to improve with more available data to learn with. 

# Next Steps

While this model has the lowest difference between training and test, loss and accuracy there is still a slight difference. In future iteration, adjusting the lambda coefficient in the l2 regularization may further decrease these differences. An additional method may also be to reduce the kernel nodes even further. 

As we have seen, the more data available to train the model, the better it performs. One option to increase data, while maintaining equal distribution of classes, is instead of down sampling the pneumonia class to match that of the normal class, we can up sample the normal class using data augmentation to match the number of pneumonia cases. This will result in 2534 more "NORMAL" images and increase the total number of training images from 2698 to 7766. 

# Recommendations

My recommendation is that GE Healthcare have built into the CCS machines software the ability to not only flag cases of concern for immediate review, but also have the radiologist confirm the AI's predictions. This collaborative approach gives further opportunities for the model to learn and improve performance, and also provides a direct actionable way to improve clinical outcomes and elevate patient experience. 

An additional recommendation would be to push software updates to existing machines, allowing for example, a machine in a small family practice in Idaho could get updates and improve due to the images being collected from clinics in more densely populated urban locations like San Francisco or New York. 

These recommendations will make these CCS units more marketable to a larger range of businesses and give every doctor’s office around the globe a reason to want this technology for their own practice. The more CCS machines learning from new cases, the better these AI predictions can get. And the better the predictions, the more justification to bring this technology into every medical practice.

# Thank You

Let's work together, 
- Email: cassigroesbeck@emailplace.com
- GitHub: [@AgathaZareth](https://github.com/AgathaZareth)
- LinkedIn: [Cassarra Groesbeck](https://www.linkedin.com/in/cassarra-groesbeck-a64b75229)