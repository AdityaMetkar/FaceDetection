# Face Detection using Deep Learning and Computer Vision
![image](https://github.com/AdityaMetkar/FaceDetection/assets/133694021/3f3dd309-707e-43bd-b7ce-3ad2ca178479)


## Folder Structure for Pre-Processing
> Create the below folder structure to implement the code
<p style="align-items:left;">
  <img src="https://github.com/AdityaMetkar/FaceDetection/assets/133694021/4cdc9e4b-a049-4c09-996c-97b51e4919ed" alt="Description of File Structure" width="350" height="350">
</p>


## Procedure
> For a detailed step-by-step Guide, follow the attached notebook

1. First we load the dataset consisting of images of cats and dogs with installing dependencies.<br>
   Libraries used are  `TensorFlow | OpenCV | Numpy | Matplotlib | Sklearn`.

2. We filter unwanted images with corrupted files or extensions and Preprocess the rest by GrayScaling them.
3. We create Labels for our dataset, in this case `[0:Cats,1:Dogs]`. This can be achieved by
   - Manual iteration of the dataset
   - Using inbuilt Keras Pipeline
4. We divide the preprocessed dataset into train, test, and validation sets to feed into our Neural Network
5. We generate the CNN Architecture using the `Sequential() API` of Tensorflow.
