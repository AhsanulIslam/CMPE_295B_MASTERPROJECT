# 2D to 3D High Resolution image conversion using generative GANs

### Objective
We exist in a three-dimensional universe but its two-dimensional (2D) representations are all we see. One of the key computer vision problems is to derive from their 2D representations the 3D forms of the objects. Initially it used 2D images to market an object. But as we are in an advanced era, where you can use 3D objects and photographs to make an object attractive. This will help to improve the company business. The picture or 3d forms that are available in the current market are of low resolution. Consequently of which the shapes and pixels often appear to be disoriented, distorted and not obvious. This project aims to create high-resolution images from the lower resolution and transforms low-dimensional probabilistic space using GANs to the space of 3D objects.

### Dataset
We used synthetic images of objects from Shapenet dataset and also real time images from Pix3D images & voxelized models for training the network, here we used subset of shapenet  dataset which consists of 50,000 3D  models and 13 major categories.From Pix3D dataste, we used the 2,895 chair images.

3D Generator Model,To train the network, we used 224 *224 RGB input image with batch sizeof 64.We implement network in PyTorch and train the model pix2vox using Adam optimizer.The initial learning rate is set to 0.0001 for first 150 epochs and then it is changed to 2.We trained the network with single view images for initial 200 epochs and then we train the entire model together with random number of  input images.

### Project Design
Our project is designed with four main modules or sub divisions namely semantic segmentation module, three dimensional reconstruction module, three dimensional super resolution module and web app.

### Project Architecture

### Results

### Performance and Analyses

### References





