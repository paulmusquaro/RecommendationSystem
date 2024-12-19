# RecommenderSystem

## Overview

This project implements a recommender system using the [Surprise](https://surprise.readthedocs.io/) library and explores matrix factorization techniques such as **SVD**, **SVD++**, and **NMF**. The **MovieLens** dataset (ml-100k) was used to train and evaluate the models, comparing their performance using metrics like RMSE (Root Mean Square Error) and MAE (Mean Absolute Error). Cross-validation and hyperparameter tuning were conducted to determine the optimal configuration for each algorithm.

### Key Features:
- Implementation of SVD, SVD++, and NMF algorithms.
- Cross-validation to evaluate models using RMSE and MAE.
- Hyperparameter optimization using GridSearchCV.
- Performance comparison across algorithms.

## Library Used

`Surprise` is a Python library created by Nicolas Hug in 2016. It was developed to simplify the creation and evaluation of recommender systems using collaborative filtering techniques. Surprise provides robust tools for implementing algorithms such as SVD, SVD++, and NMF, along with utilities for cross-validation and hyperparameter optimization. The library also supports built-in datasets like MovieLens, making it an excellent tool for research and experimentation.

## Results

The following algorithms were evaluated using the MovieLens dataset:

1. **SVD**:
   - Average RMSE: ~0.9365
   - Average MAE: ~0.7379
   - Moderate computation time.

2. **SVD++**:
   - Average RMSE: ~0.9206
   - Average MAE: ~0.7224
   - Best overall accuracy but high computation time.

3. **NMF**:
   - Average RMSE: ~0.9632
   - Average MAE: ~0.7569
   - Faster computation but less accurate.

#### Conclusion:
- **SVD++** provides the best accuracy for RMSE and MAE, making it ideal for this dataset.
- For larger datasets or limited computational resources, **SVD** is recommended due to its lower computation time.


## Conda (Setup and Environment)

To make the project reproducible and ensure smooth package management, this project uses Conda as a package and environment manager. Below are the steps to set up the environment:


1. **Install Conda**:
If you haven't installed Conda yet, you can download it from the official [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) websites. Anaconda is a larger distribution with more pre-installed packages, while Miniconda is a smaller, minimal version. Choose whichever suits your needs.

2. **Create a new environment:** Open your terminal and run the following command to create a new Conda environment with Python 3.12:

    ```bash
    conda create --name new_conda_env python=3.12
    ```

3. **Activate the environment:** Once the environment is created, activate it by running:

    ```bash
    conda activate new_conda_env
    ```

4. **Install required packages (Jupyter, NumPy, MatPlotLib, Pandas, Scikit-Learn and Surprise)**

    ```bash
    conda install jupyter numpy matplotlib pandas scikit-learn
    conda install -c conda-forge scikit-surprise
    ```

5. **Run Jupyter Notebook**

    ```bash
    jupyter notebook
    ```



## Summary

This project demonstrates the power of matrix factorization techniques in building recommendation systems. By leveraging the Surprise library, it is possible to efficiently train and evaluate models and make data-driven decisions about the best-performing algorithms. The results highlight that model selection depends on the trade-off between accuracy and computational cost.