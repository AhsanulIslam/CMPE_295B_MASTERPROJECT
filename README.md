# 2D to 3D High Resolution image conversion using generative GANs

### Objective
We exist in a three-dimensional universe but its two-dimensional (2D) representations are all we see. One of the key computer vision problems is to derive from their 2D representations the 3D forms of the objects. Initially it used 2D images to market an object. But as we are in an advanced era, where you can use 3D objects and photographs to make an object attractive. This will help to improve the company business. The picture or 3d forms that are available in the current market are of low resolution. Consequently of which the shapes and pixels often appear to be disoriented, distorted and not obvious. This project aims to create high-resolution images from the lower resolution and transforms low-dimensional probabilistic space using GANs to the space of 3D objects.

### Dataset
We used synthetic images of objects from Shapenet dataset and also real time images from Pix3D images & voxelized models for training the network, here we used subset of shapenet  dataset which consists of 50,000 3D  models and 13 major categories.From Pix3D dataste, we used the 2,895 chair images.

3D Generator Model,To train the network, we used 224 *224 RGB input image with batch sizeof 64.We implement network in PyTorch and train the model pix2vox using Adam optimizer.The initial learning rate is set to 0.0001 for first 150 epochs and then it is changed to 2.We trained the network with single view images for initial 200 epochs and then we train the entire model together with random number of  input images.


   - ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
   - ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
   - Pix3D images & voxelized models: http://pix3d.csail.mit.edu/data/pix3d.zip

### Project Design
Our project is designed with four main modules or sub divisions namely super resolution module,semantic segmentation module, three dimensional reconstruction module and web app.

![Project Pipeline](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/pipeline.png)

##### 1. Super Resolution Module

Single Super-resolution image (SISR), tries to capture a high-resolution (HR) image from a single low-resolution (LR) picture. The basic model is constructed with residual blocks and optimized using perceptual loss in a GAN framework. Nevertheless a major gap still exists between SRGAN findings and ground-truth (GT) images.

##### 2. Semantic Segmentation Module

The software DeepLab V3 + is used for the method of semantine segmentation. The semantic segmentation distinguishes the target from the background in the image and transforms the 3-channel RGB image into a 4-channel RGBA image. The RGBA image is fed into the architecture for the 3D reconstruction.

##### 3. 3D Reconstruction Module

The 3D Generator subsystem architecture is built for reconstruction of single and multi views. Module is designed to recreate the 3D form of an object, either from single or multiple RGB images. The 3D shape of an entity is represented by a 3D voxel map, where 0 is an empty cell, and 1 is an occupied cell. Framework has four main modules which are encoder, decoder, context-conscious fusion and refiner.

##### 4. Web App

Streamlit application is used for developing the front end application. Where user uploads the 2d rgb image, it passes through super resolution model and generates the semantic segmented image. This when processed finally outputs the 3D voxeled respresentation image.

### Project Architecture
![Project Architecture](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/Project%20Architecture.png)

### Model Path

1. Super Resolution Model: s3://2dto3d/superres/RRDB_ESRGAN_x4.pth
2. 2D to 3D Model: s3://2dto3d/2d_to_3d_pretrainedmodel.pth

### Results
User uploads the input 2D low resolution image.
![Uploaded Image](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/Car%20Uploaded%20Images.png)

Super resolution image is generated using ESRGAN
![Super Resolution Image](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/SR.png)

Super resolution image is segmented 
![Segementation Image](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/Segmented%20Image.png)

Model outputs 3D super resolution voxel image
![3D](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/3D.png)

### Performance and Analyses

Below image depicts model performance for different domain
![Model Results](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/comparision_results.png)

Comparision analyses of existing models with our model
![Model Comparision](https://github.com/Radhika009/CMPE_295B_MASTERPROJECT/blob/master/Images/model_comparision.png)

### References

1. https://arxiv.org/pdf/1901.11153.pdf
2. https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf
3. https://arxiv.org/pdf/1604.00449v1.pdf
