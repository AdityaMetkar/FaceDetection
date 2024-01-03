# Face Detection using Deep Learning and Computer Vision
![image](https://github.com/AdityaMetkar/FaceDetection/assets/133694021/cdf97f9a-ff96-4033-98f6-49ea8c36ab27)


## Folder Structure for Pre-Processing
> Create the below folder structure to implement the code
<p style="align-items:left;">
  <img src="https://github.com/AdityaMetkar/FaceDetection/assets/133694021/4cdc9e4b-a049-4c09-996c-97b51e4919ed" alt="Description of File Structure" width="350" height="350">
</p>

## Procedure
> For a detailed step-by-step Guide, follow the attached notebook

1. First we load the dataset consisting of images of cats and dogs with installing dependencies.<br>
   Libraries used are  `TensorFlow | OpenCV | Numpy | Matplotlib | Sklearn`.

2. We create our Facial Images dataset by using the Webcam.
3. The Next Step is to create labels for these saved images. I have used the `LabelMe` library for this purpose.
   - For Each Image present in the Dataset, we create a corresponding set of Rectangular Coordinates (Label)
   - This info is saved in a JSON file for later access during the training phase.
     <p style="align-items:left;">
      <img src="https://github.com/AdityaMetkar/FaceDetection/assets/133694021/545de12d-c99e-4b0e-ac6b-4bfa43973810" alt="Sample Conversion"  height="350">
     </p>
   - Now we have a dataset with images and corresponding JSONs with coordinates.
4. We divide the dataset into train, test, and validation Sets.
5. On Each of these Sets(Folders) we apply Image Augmentation to increase the dataset
   - For this purpose, we will use the `Albumentations` Library.
   - We apply augmentations like
     <br><br><table>
        <tr>
          <td>
            - Random Crop <br>
            - Horizontal Flip <br>
            - Random Brightness Contrast <br>
            - Random Gamma <br>
            - RGBShift <br>
            - VerticalFlip
          </td>
          <td style="justify-content:center;">
            <img src="https://github.com/AdityaMetkar/FaceDetection/assets/133694021/44c59e3c-8781-4fc2-8dd9-15fd5847f284" alt="Sample Conversion" width="50%">
          </td>
        </tr>
     </table>
   - All the augmented images and labels will be saved in their respective folders in the `augmented Folder`.
     
6. We generate our Deep Learning Model using the `Functional() API` of Tensorflow.
    - Model Summary<br><br>
      > Note: Here there are two output layers 
      >  1) Dense(1) for the Binary Classification
      >  2) Dense(4) for the Coordinates
       <br>
       <table>
          <tr>
              <th>Layer</th>
              <th>Operation</th>
              <th>Output Shape</th>
          </tr>
          <tr>
              <td>Input</td>
              <td>Input(shape=(120, 120, 3))</td>
              <td>(120, 120, 3)</td>
          </tr>
          <tr>
              <td>VGG16</td>
              <td>VGG16(include_top=False)</td>
              <td>Output from VGG16</td>
          </tr>
          <tr>
              <td>GlobalMaxPooling2D</td>
              <td>GlobalMaxPooling2D()</td>
              <td>(Depth of VGG16 Output)</td>
          </tr>
          <tr>
              <td>Dense</td>
              <td>Dense(2048, activation='relu')</td>
              <td>(2048)</td>
          </tr>
          <tr>
              <td>Dense</td>
              <td>Dense(1, activation='sigmoid')</td>
              <td>(1)</td>
          </tr>
          <tr>
              <td>GlobalMaxPooling2D</td>
              <td>GlobalMaxPooling2D()</td>
              <td>(Depth of VGG16 Output)</td>
          </tr>
          <tr>
              <td>Dense</td>
              <td>Dense(2048, activation='relu')</td>
              <td>(2048)</td>
          </tr>
          <tr>
              <td>Dense</td>
              <td>Dense(4, activation='sigmoid')</td>
              <td>(4)</td>
          </tr>
      </table>
    - We define Custom Losses
      - Classification Loss (BinaryCrossEntropy)
      - Localization Loss (bbox Loss)
7. We compile the model with the custom losses and optimizers and fit it with the training and validation data
8. We plot and analyze the Performance Metrics and Save the Model.

## Testing the Model using WebCam
We can load the saved model and check out its Real-Time capabilities
![image](https://github.com/AdityaMetkar/FaceDetection/assets/133694021/f2564ca1-c0c8-40ed-8708-e1c9e815a82f)

