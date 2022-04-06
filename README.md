# Handwritten-Equation-Solver
①PyQt for canvas and handwritten equation input, 

②crop digits and operators from input, 

③MobileNet for recognition,

④Python build-in function eval() to calculate string to output result.

## Requiments
PyQT5

torch

torchvision

PIL

opencv

tqdm(Used in train.py, disable it in program if not intending to show progress bar.)

## Overview Process
![image](https://user-images.githubusercontent.com/61113791/161964590-79f4380b-b241-441f-9ff7-1df7e5acf03e.png)

### Input HandWritten Equations
Write it in canvas and click "Exit" button, then "save test.jpg" will be saved in your current directory. 
![image](https://user-images.githubusercontent.com/61113791/161966141-3461b580-1ab5-4b63-a115-52fd971e5131.png)



### Crop digits and operators
Then it will be processed by get_num.py to crop the digits and operators, and packed in a batch of tensors.

Crop based on the black pixels, but it is not a ingenious method, which requires digits and operators not overlapped and written wholely.

You can change the method using CV2.FindContours functions, following methods in Links for more.
![image](https://user-images.githubusercontent.com/61113791/161967125-d8a581e4-6cc1-4847-b781-a405fc828bd2.png)

### Recognition and calculation
#### Train the model
We train a MobileNetV2 to recognize the 14 (0~9 and +-×÷) categories.

They are collected in folders named after the digits, ./0/0_XXX.jpg for example, and folders for operators are in {10for+ 11for- 12for× 13for÷} manner.

Since the input images are grayscale, we change input of the
```
features = [ConvBNReLU(3, input_channel, stride=2)]
```
to
```
features = [ConvBNReLU(1, input_channel, stride=2)]
```
Nothing else but num_classes(defualt 1000 -> 14) is changed.

We train MobileNetV2 to perofrom the Handwritten Equation Solver on mobile platform, train your own models as you wish.

***It should be noted that our model unfortunately does not perform well on recognizing the digits and operators, don't be suprised if it fails you :-)***

### Index to String to Float Results
The model outputs its recognition of input characters in torch.tensor

We transform the prediction from a batch of tensors to string.

Then the string are calculated through Python build-in function eval().

```
tensor(1.)  tensor(10.) tensor(1.)

string("1+1")

out = eval("1+1") 
out ---> 2.
```

Check try04.py for detailed code.

## Dataset
https://www.kaggle.com/xainano/handwrittenmathsymbols
We utilize the digits 0~9 and operators +-×÷ of it.

## Links
https://www.geeksforgeeks.org/handwritten-equation-solver-in-python/

https://github.com/sabari205/Equation-Solver

https://github.com/matheuscr30/Handwritten-Equation-Recognition---CNN

