# BENCHMARKED INSIGHTS CALCULATOR FOR CNNs/DNNs

### By Nilaksh Agarwal (NA2886), Chandan Suri (CS4090)

#### Course project for COMSE6998_012_2021 - RACTICAL DEEP LEARNING SYSTEM PERFORMANCE under Prof. Parijat Due

## Motivation:

1. Very hard to gauge training time for a model based on the model parameters & GPU.

2. Cost & Availability of GPUs is a big concern which can be resolved using a benchmarking calculator!

3. Correlation between model parameters is mostly an unknown for different CNNs and FC networks.

![alt text](Pipeline.png)

## Repository 

```
AllRunsv2.0.ipynb
```
Creating the different models, running the different models, and saving the running time as pickle files.


```
Analysis_v4_0.ipynb
```
Analyzing the various results, plotting the correlational graphs etc.


```
UI_v2.0.ipynb
```
The final notebook to run the UI and the benchmarking calculator.


## Results

### Models generated

VGG Model
![alt text](src/models/vgg.png)

Resnet Model
![alt text](src/models/resnet.png)

Inception Model
![alt text](src/models/inception.png)

Fully Connected Model
![alt text](src/models/fc.png)
