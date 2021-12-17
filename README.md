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


### Analysis 

VGG Analysis 

![alt text](Visualizations/VGG_data_vs.png)

![alt text](Visualizations/VGG_aux_vars_data_vs.png)

ResNet Analysis 

![alt text](Visualizations/ResNet_data_vs.png)

![alt text](Visualizations/ResNet_aux_vars_data_vs.png)

Inception Analysis 

![alt text](Visualizations/Inception_data_vs.png)

![alt text](Visualizations/Inception_aux_vars_data_vs.png)

<!-- Fully Connected Analysis 

![alt text](Visualizations/Fully\ Connected_aux_vars_data_vs.png)

![alt text](Visualizations/Fully\ Connected_aux_vars_data_vs.png) -->

### Demo

![alt text](vgg_demo.png)

![alt text](resnet_demo.png)

![alt text](inception_demo.png)

![alt text](fc_demo.png)

## Conclusions

1. Based on the results we got, we can successfully model the training time and cost based on model parameters provided by the user without no memory and disk considerations. 

2. Also, we could successfully find correlations between various model parameters that could be mapped to the training time and costs.
This can be quite easily extended to multiple GPUs to create a larger database of the model parameterizations vs. training time and costs. (if we get the GPUs!)

3. Also, this can be extended to include multiple cloud providers to create a complete database of models vs. model configurations vs. GPUs vs. cloud providers that could help a lot of people to find which all configurations to consider without incurring any costs.

4. Such a benchmarking tool can also help many companies in finding the best approach for their projects and thus, save a lot of time and resources.
