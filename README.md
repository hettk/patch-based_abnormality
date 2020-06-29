# Patch-based abnormality


Implementation of MICCAI 2020, Hett et al. 2020

## Abstract
<p align="justify">Deep learning techniques have demonstrated state-of-the-art performances in many medical imaging applications. These methods can efficiently learn specific patterns. An alternative approach to deep learning is patch-based grading methods, which aim to detect local similarities and differences between groups of subjects. This latter approach usually requires less training data compared to deep learning techniques. In this work, we propose two major contributions: first, we combine patch-based and deep learning methods. Second, we propose to extend the patch-based grading method to a new patch-based abnormality metric. Our method enables us to detect localized structural abnormalities in a test image by comparison to a template library consisting of images from a variety of healthy controls.  We evaluate our method by comparing classification performance using different sets of features and models. Our experiments show that our novel patch-based abnormality metric increases deep learning performance from 91.3% to 95.8% of accuracy compared to standard deep learning approaches based on the MRI intensity</p>



<p align="center"><img src="figures/pipeline.png" width="600"><br>
Pipeline of the proposed method. First, HC from the dataset is separated into two subset, the HC templates used to estimate the local abnormality, the second is the set of HC for the evaluation of our method. Once all MRIs are preprocessed, we estimate the local abnormality using the HC template library. Finally, a convolutional neural network with softmax is used to obtain final classification.</p>

## Patch-based abnormality index


<p align="center"><img src="figures/pbd_illustration.png" width="600"><br>
Illustration of patch-based abnormality maps for **Top** a healthy control subject and **Bottom** an HD patient with 40 CAG repeats. **From left to right**, 3 different time points are shown for each subject. The HD subject is in the pre-manifest stage for the first time points, but converts to clinical diagnosis by the third time point. **Blue** represents areas with a low abnormality score *a(x)*, whereas **Red** represents areas with high abnormality score *a(x)*. We note a progressive increase of abnormality near the basal ganglia during the course of the disease which is consistent with HD pathology, while the abnormality map for the HC subject remains stable.</p>

## Deep-learning classification
<p align="center"><img src="figures/network.png" width="600"><br>
Illustration of the convolutional neural network architecture used to validate our work. The architecture consist of a combination of convolutional layer (Conv), batch normalization (BN), Skipped connection layer (Side), pooling layer (Pool), and fully connected layer (FC). A soft-max layer estimate the probability for each class.</p>
