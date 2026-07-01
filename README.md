# Turbofan Engine Predictive Maintenance

## Project Overview

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS dataset. The goal is to estimate how many operational cycles an engine has remaining before failure, enabling predictive maintenance and reducing unexpected breakdowns.

## Objectives

* Predict Remaining Useful Life (RUL) of aircraft engines.
* Compare Machine Learning and Deep Learning models.
* Develop a foundation for a maintenance decision-support system.
* Improve maintenance planning using predictive analytics.

## Dataset

* **Dataset:** NASA C-MAPSS
* **Subset Used:** FD001
* **Target Variable:** Remaining Useful Life (RUL)

## Project Workflow

1. Data loading
2. Data preprocessing
3. Feature selection
4. Data normalization
5. Sequence generation for LSTM
6. Model training
7. Model evaluation
8. Performance comparison
9. Predictive maintenance decision support

## Models Implemented

* Linear Regression
* Random Forest Regressor
* Long Short-Term Memory (LSTM)

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

## Repository Structure

```text
Turbofan-Engine-Predictive-Maintenance/
│
├── data/
├── docs/
├── images/
├── models/
├── notebooks/
├── results/
├── src/
├── README.md
├── LICENSE
└── .gitignore
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras

## Future Work

* Improve LSTM performance
* Implement a maintenance decision-support system
* Explore Bidirectional LSTM and other deep learning architectures
* Compare performance across additional C-MAPSS datasets

## Author

**Ritik Tickoo**

Master of Business Analytics
Deakin University
