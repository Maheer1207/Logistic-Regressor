# LogiBoost: Optimized Logistic Regression for Large-Scale Datasets

## Project Overview

LogiBoost is a machine learning (ML) project that implements a highly optimized logistic regression model designed for large-scale datasets. In this project, the model is trained on a dataset $$\( D = \{(X_i, Y_i)\}_{i=1}^n \)$$ consisting of $$\( n \)$$ samples, where each feature vector $$\( X_i \in \mathbb{R}^d \)$$ and the corresponding label $$\( Y_i \in \{0, 1\} \)$$. The conditional expectation of $$\( Y_i \)$$ given $$\( X_i \)$$ is modeled using the logistic (sigmoid) function:

$$
E[Y_i | X_i] = \frac{1}{1 + \exp(-X_i^\top \theta^*)}
$$

Here, $$\( \theta^* \in \mathbb{R}^d \)$$ represents the true parameter vector that governs the underlying data distribution.

### Project Objectives

The primary goal of LogiBoost is to implement an ML pipeline that efficiently estimates the parameter vector $$\( \theta \)$$ for a logistic regression model, using a dataset of size $$\( n = 1,000,000 \)$$ (provided in `feature.npy` and `obs.npy`). The estimated parameter vector $$\( \theta \in \mathbb{R}^{10} \)$$ should be a close approximation of the true vector $$\( \theta^* \)$$.

### Key Achievements

LogiBoost has successfully met its objectives by delivering:

1. **High Computational Efficiency:** The ML pipeline executes in under 5 seconds on a Google Colab environment, demonstrating its scalability for large datasets.
2. **Precision in Parameter Estimation:** The resulting parameter vector $$\( \theta \)$$ closely approximates the true parameter vector $$\( \theta^* \)$$, effectively minimizing the logistic loss on the dataset.

### Technical Approach

LogiBoost leverages advanced optimization techniques within the ML framework to minimize the expected logistic loss function. The `scipy.optimize.minimize` function is utilized to efficiently find the optimal parameter vector $$\( \theta \)$$.

#### Objective Function

The logistic loss, also known as binary cross-entropy, is employed as the objective function. This is a standard loss function in binary classification tasks in ML. The Jacobian (gradient) of the loss function is computed to provide gradient information for the optimization algorithm.

#### Optimization Strategy

LogiBoost employs the `'L-BFGS-B'` algorithm, a quasi-Newton method particularly well-suited for large-scale ML problems. This algorithm efficiently handles the optimization of the logistic loss function by approximating the Hessian matrix, making it ideal for high-dimensional parameter spaces.

### Model Evaluation

The performance of the logistic regression model was evaluated by applying the optimized parameter vector $$\( \theta \)$$ to predict the outcomes $$\( Y_i \)$$ for the provided feature set. The model's accuracy was calculated by comparing the predicted labels against the actual labels in `obs.npy`, yielding a high classification accuracy that meets the project benchmarks.

## Installation

To execute LogiBoost, ensure your environment has Python installed along with the following essential ML libraries:

```bash
pip install numpy scipy
```

## Usage Guide

To use LogiBoost, follow these steps:

1. **Open the Jupyter Notebook:**
   - In your terminal, navigate to the root directory of the project where `logistic_regressor.ipynb` is located.
   - Launch Jupyter Notebook by running:
     ```bash
     jupyter notebook
     ```
   - In the Jupyter interface, open `logistic_regressor.ipynb`.

2. **Load the Dataset:**
   - The notebook is configured to automatically load `features.npy` and `obs.npy`. Ensure these files are located in the same directory as the notebook.

3. **Explore the Code:**
   - The notebook is organized into sequential cells with detailed explanations. Review the markdown cells to understand each step of the ML process.
   - The logistic regression model and optimization process are implemented in Python using the `scipy.optimize.minimize` function.

4. **Run the ML Pipeline:**
   - Execute each cell in the notebook to run the full logistic regression pipeline.
   - The notebook will:
     - Load and preprocess the dataset.
     - Define and optimize the logistic regression model.
     - Output the optimized parameter vector $$\( \theta \)$$ and the model’s accuracy.

5. **Analyze Results:**
   - Upon completion, review the final output in the notebook, which includes the optimized parameter vector $$\( \theta \)$$ and the accuracy metrics of the model.

6. **Experiment and Extend:**
   - Modify the notebook to experiment with different optimization algorithms, regularization techniques, or dataset sizes to further explore logistic regression and ML optimization.

## Contribution

We welcome contributions to enhance the LogiBoost project. Feel free to fork the repository, create a feature branch, and submit a pull request for review.
