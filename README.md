# Classification-Neural-Network
### Overview
In this project, we are given a dataset containing the images of dogs and cats. The images in the dataset are divided for training and testing. The goal of this project is to create a convolutional neural network which can learn to recognize an image and say whether it is dog or a cat. 

### Softwares Used
1. Python 3
2. Jupyter Notebook
3. Google Colab

### Libraries Used
1. Pytorch
2. torch vision

### Dataset
The data set used for neural net can be found [here](https://drive.google.com/file/d/19inwa0n1W4DZamjCOm5XAlztvqG_xkjP/view)

### Steps to run the code
1. Download the datasets from the above link.

2. Clone the repository
```
git clone https://github.com/mesneym/Classification-Neural-Network.git
cd code
```

3. Open google colab using this [link](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

4. Open classifier.ipynb file in code fole in google colab.

5. Now move the dataset folder to the directory where clssifier.ipnyb file is stored.

6. Start executing the cell one by one.

The first cell is used to connect the drive to google collab.
```
from google.colab import drive
drive.mount('/content/drive')
```
