# FacialKeyDetection

This is a model based on detecting 15 keypoints in a human face.
The input to the model is an image containing a human-face and the outputs are 15 set of coordinates 
depicting the keypoints such as left_eye, right_eye, upperr_lip, lower_lip etc.
The dataset is downloaded from Kaggle and is found in the link below.
https://www.kaggle.com/c/facial-keypoints-detection

About 50% of the images are not labelled completely. 
So, these unlabelled images have been dropped and the labelled images are augmented with techniques 
such as random rotation and horizontal flip.
