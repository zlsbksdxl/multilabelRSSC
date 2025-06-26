# CSSFE

This repository provides the code for the method in our paper '**Category-Specific Selective Feature Enhancement for Long-Tailed Multi-Label Image Classification**'. (ICCV2025 accept)



**If you have any questions, you can send me an email. My mail address is drq15145136147@gmail.com.**
# Abstract
Since real-world multi-label data often exhibit significant  label imbalance, long-tailed multi-label image classification has emerged as a prominent research area in computer vision. Traditionally, it is considered that deep neural networks’ classifiers are vulnerable to long-tailed distri-butions, whereas the feature extraction backbone remains relatively robust. However, our analysis from the feature learning perspective reveals that the backbone struggles to maintain high sensitivity to sample-scarce categories but retains the ability to localize specific areas effectively. Based
on this observation, we propose a new model for long-tailed
multi-label image classification named category-specific selective feature enhancement (CSSFE). First, it utilizes the
retained localization capability of the backbone to capture
label-dependent class activation maps. Then, a progressive
attention enhancement mechanism, updating from head to
medium to tail categories, is introduced to address the low confidence issue in medium and tail categories. Finally, visual features are extracted according to the optimized class
activation maps and combined with semantic information
to perform the classification task. Extensive experiments on
two benchmark datasets highlight our findings’ generalizability and the proposed CSSFE’s superior performance.


![本地路径](model.jpg )
![本地路径](fla.jpg )







