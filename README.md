# Deep Learning 1470: Final Project


## What’s in Your Brain?: A CNN Approach to Tumor Detection
Daniel Cho, Kameel Dossal, David Han, Erica Song


## Introduction
Convolutional neural networks are a deep learning architecture commonly used for image classification tasks. In the healthcare field, medical professionals use various imaging techniques on the body to properly diagnose patients with diseases. Brain tumors represent one area where machine learning has the potential to help healthcare professionals make more informed decisions, saving time and money. In this project, we implemented a CNN model that classifies brain tumors using MRI images. The categories of tumors were glioma, meningioma, pituitary, and no tumor. We chose to emulate the 2019 paper “Brain Tumor Classification Using Convolutional Neural Network” by Das et al. because we wanted to implement multi-class classification of images. This model also serves an important purpose since survival rate depends on the type of tumor, and this model can help doctors correctly classify tumors and provide the most fitting treatment to their patients in the early stages of the disease.


`preprocess.py:` Used to preprocess original Kaggle Data
`model.py:` Our CNN Model
`main.py:`  Trains and Tests our Model

`explore.ipynb:`     Used to generate confusion matrix and test our model
`final_weights.h5:`  Model Weights for the final trained model
`lime_explainer.py:` Generates images and content for LIME explainer

