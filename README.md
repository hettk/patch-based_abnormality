# Patch-based abnormality


Implementation of MICCAI 2020, Hett et al. 2020


### Summary
- PatchBasedAbnormality
    - main function : pba.m
- DeepLearningClassification
    - train.py, test.py



## Abstract
<p align="justify">Deep learning techniques have demonstrated state-of-the-art performances in many medical imaging applications. These methods can efficiently learn specific patterns. An alternative approach to deep learning is patch-based grading methods, which aim to detect local similarities and differences between groups of subjects. This latter approach usually requires less training data compared to deep learning techniques. In this work, we propose two major contributions: first, we combine patch-based and deep learning methods. Second, we propose to extend the patch-based grading method to a new patch-based abnormality metric. Our method enables us to detect localized structural abnormalities in a test image by comparison to a template library consisting of images from a variety of healthy controls.  We evaluate our method by comparing classification performance using different sets of features and models. Our experiments show that our novel patch-based abnormality metric increases deep learning performance from 91.3% to 95.8% of accuracy compared to standard deep learning approaches based on the MRI intensity</p>

<br>

<p align="center"><img src="figures/pipeline.png" width="600"><br>
Pipeline of the proposed method. First, HC from the dataset is separated into two subset, the HC templates used to estimate the local abnormality, the second is the set of HC for the evaluation of our method. Once all MRIs are preprocessed, we estimate the local abnormality using the HC template library. Finally, a convolutional neural network with softmax is used to obtain final classification.</p>
<br>

## Patch-based abnormality index

Our method derives from the patch-based grading framework. To address PBG's dependence on the two template libraries, we estimate the local differences from a single template library composed only of HC subjects (see Fig.~\ref{fig:pipeline}). The abnormality $a(x)$ for each voxel $x$ of the MRI under study, is defined as:
 
$a(x) = \frac{\sum_{T \in K_x} ||S(x) - T(y)||_2^2 }{\sigma_{S(x)}},$

where $\sigma_{S(x)}$ is standard deviation of intensities over the patch $S(x)$, which normalizes the differences of signal intensity contained in each patch $S(x)$. Similar to Eq.\ 1, $K_x$ is the set of closest patches provided by the PatchMatch algorithm.  This results in a low abnormality metric if the current patch is similar to age-matched control subjects, and in a high abnormality metric if the patch does not fit well within the distribution of age-matched control subjects. 
<br>

<p align="center"><img src="figures/pbd_illustration.png" width="600"><br>
  Illustration of patch-based abnormality maps for <b>Top</b> a healthy control subject and <b>Bottom</b> an HD patient with 40 CAG repeats. <b>From left to right</b>, 3 different time points are shown for each subject. The HD subject is in the pre-manifest stage for the first time points, but converts to clinical diagnosis by the third time point. <b>Blue</b> represents areas with a low abnormality score <i>a(x)</i>, whereas <b>Red</b> represents areas with high abnormality score <i>a(x)</i>. We note a progressive increase of abnormality near the basal ganglia during the course of the disease which is consistent with HD pathology, while the abnormality map for the HC subject remains stable.</p>

## Deep-learning classification

In order to model the spatial disease signature and perform the subject-level classification, we used a convolutional neural network (CNN) approach. In recent years, many different architectures have been proposed in the pattern recognition field. Among them, deep residual neural network (ResNet) has shown competitive performances \cite{he2016deep}. This architecture is characterized by skipped connections of different blocks of layers (see Fig.~\ref{fig:HHC}). ResNet has demonstrated a reduced training error compared to other networks with similar depth. Indeed, the residual mapping enables to reduce the training error, which is generally correlated with the network depth for classic stacked architectures. In addition, to address the problem of GPU memory limitation, we used a 3D patch approach. Thus, both networks have as input 8 channels that represent non-overlapping patches from the input data (\emph{i.e.}, T1w MRI or PBA maps).

<p align="center"><img src="figures/network.png" width="600"><br>
Illustration of the convolutional neural network architecture used to validate our work. The architecture consist of a combination of convolutional layer (Conv), batch normalization (BN), Skipped connection layer (Side), pooling layer (Pool), and fully connected layer (FC). A soft-max layer estimate the probability for each class.</p>


## References
[1] he2016deep
[2] hettk
[3] SNIPE
