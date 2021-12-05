# Dog Breed Classifier
I tuned different Pytorch/Tensorflow pre trained models like ResNet50 , Wide ResNet_50.2 , VGG16 and a custom CNN model to classify a dog image among 120 breeds.
The training dataset can be collected from [here](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar).

## Results :

### Wide ResNet_50.2 :
Test Accuracy :   82.88978494623656   
Test Loss     :   0.6641809101546964
- #### Loss :
  ![wide_resnet50_loss](https://user-images.githubusercontent.com/57902078/137465987-cee6d0ae-923e-42c2-81d7-a83b59e09d77.png)
- #### Accuracy :
  ![wide_resnet50_accuracy](https://user-images.githubusercontent.com/57902078/137466085-4056e779-73be-4d57-ab8b-c29e6e46cf60.png)

### ResNet50 :
Test Accuracy :   80.26881720430107 / 0.9375 (Tensorflow)
Test Loss :       0.6438697055783323 / 0.9062 (Tensorflow)
- #### Loss :
  ![resnet_50_loss](https://user-images.githubusercontent.com/57902078/137466849-3392854a-9d1e-463a-b8d2-645303d5f7f1.png)

- #### Accuracy :
  ![resnet_50_accuracy](https://user-images.githubusercontent.com/57902078/137466887-f871777c-0276-4adb-a78e-68d70c42422e.png)

### VGG16 :
Test Accuracy :   80.8736559139785      
Test Loss :       0.6329619951385964
- #### Loss :
  ![vgg16_loss](https://user-images.githubusercontent.com/57902078/137476625-6473d98d-0e73-47ac-8a36-d5810c3d7039.png)

- #### Accuracy :
  ![vgg16_accuracy](https://user-images.githubusercontent.com/57902078/137476645-fb5c7f00-ab01-49f6-bfd8-dc3b3614c6b6.png)

### custom_CNN :
Test Accuracy :   1.1548913043478262  
Test Loss :       4.78740421585414 
- #### Loss :
  ![custom_CNN_loss](https://user-images.githubusercontent.com/57902078/137493266-ebda15a8-753d-4c6b-bd54-61013e6d106b.png)
- #### Accuracy :
  ![custom_CNN_accuracy](https://user-images.githubusercontent.com/57902078/137493380-6bbc1071-f5b0-41da-b3f5-f3c093b4b2fb.png)
