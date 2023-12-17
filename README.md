# Adversarial-Attack-for-Bird-species-classification
![](./readme_img/testset.png)
* This is the classification task with the adversarial-attack issue.
## Adversarial Attack Description:
An adversarial example is an input to a machine learning model that is purposely designed to cause a model to make a mistake in its predictions despite resembling a valid input to a human.

![](./readme_img/adversarial.png)

* Image is taken from [here](https://towardsdatascience.com/breaking-neural-networks-with-adversarial-attacks-f4290a9a45aa)


## Dependencies
```shell
pip install keras
```
## Datasets
* Images in the dataset are stored in category folders
  ![](./readme_img/dataset.png)

* Found 43622 images belonging to 300 classes inside the training dataset.

|Dataset|train|Valid|Test|
|:--:|:--:|:--:|:--:|
|Image num|43622|1500|3244|
## Code
#### train.py
```shell
  python train.py
```
* Adding the noise on the images to make them look like they have been adversarial attacks.
* Augmentation of the images not only with adding noise but also other common augmentation methods to make the model more robust.

#### Inference.py

## Experiment Result
