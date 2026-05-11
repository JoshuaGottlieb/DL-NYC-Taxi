# Deep Learning Using the NYC Yellow Taxi Dataset

## Summary

This project uses the NYC Yellow Taxi data from January, March, and May of 2020 to explore the uses of neural networks in predicting total trip time for individual rides based on numerous factors. These time periods were used because the periods cover pre- and post-COVID-19 shutdown. Taxi usage patterns changed dramatically during the pandemic, which causes problems for modeling, as models trained on older datasets may perform poorly on newer datasets with different underlying data distributions. This phenomenon is known as data drift, and being able to identify and correct for data drift is essential for modern data science in order to maintain relevant and useful models.

The project is split into two parts. The first part is an extensive exploratory data analysis focusing on data cleaning, data drift detection, and feature selection. The second part of the project focuses on the use of Tensorflow and PyTorch to create simple sequential neural networks to predict taxi trip time. Three different model types were tested, along with a variety of optimizers and learning rates, resulting in 27 different combinations of network architecture, learning rate, and optimizer. A thorough analysis was conducted to gauge the effectiveness of each combination, including ability to converge, smoothness of convergence, overfitting, speed of training in raw time and number of epochs, and lowest error.

## Requirements

The libraries and version of Python used to create this project are listed below. The requirements are also available at [requirements.txt](https://github.com/JoshuaGottlieb/DL-NYC-Taxi/blob/main/requirements.txt).

```
keras==3.11.3
matplotlib==3.10.7
numpy==2.3.3
pandas==2.3.3
scikit_learn==1.7.2
scipy==1.16.2
seaborn==0.13.2
tensorboard==2.17.1
tensorflow==2.17.1
tensorflow_data_validation==1.17.0
tensorflow_metadata==1.17.2
torch==2.9.0+cpu
```

## Repository Structure

```
├── data                                           # Raw and processed NYC Taxi data, weather data, and taxi zone lookups
├── logs                                           # Training and validation metrics across training epochs
├── models                                         # Model weights at each epoch of training
├── predictions                                    # Model predictions and classification metrics
├── src                                            # Project notebooks and source code
│   ├── EDA_NYC_Yellow_Taxi.ipynb                      # Notebook containing extensive EDA analysis
│   ├── NYC_Yellow_Taxi_Modeling.ipynb                 # Notebook containing Tensorflow and PyTorch modeling and evaluation analysis
│   ├── modules                                        # Source code with custom functions
│   │   ├── plotting.py                                    # Functions for Matplotlib and Seaborn plotting
│   │   ├── plotting_utils.py                              # Helper functions for plotting
│   │   ├── preprocessing.py                               # Functions for preprocessing raw datasets
│   │   ├── statistics.py                                  # Functions for EDA statistical techniques
│   │   ├── tfdv_utils.py                                  # Helper functions for using the Tensorflow Data Validation library
│   │   ├── training.py                                    # Functions for training. logging, and saving models under PyTorch and Tensorflow
│   │   └── utils.py                                       # Functions for loading training metrics and times
├── README.md
└── requirements.txt
```
