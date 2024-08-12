# Logistic Regression Implementation

## Project Description

This project implements a machine learning (ML) model using logistic regression. The learner is presented with a dataset \( D = \{(X_i, Y_i)\}_{i=1}^n \) of size \( n \). Here, \( X_i \in \mathbb{R}^d \) is a \( d \)-dimensional feature vector and \( Y_i \in [0, 1] \) is a binary random observation. The expected value of an observation \( Y_i \) conditioned on a feature vector \( X_i \) is modeled as:

\[
E[Y_i | X_i] = \frac{1}{1 + \exp(-X_i^\top \theta^*)}
\]

where \( \theta^* \in \mathbb{R}^d \) is the \( d \)-dimensional true parameter vector that generates the observations.

The goal of this project was to implement ML code that takes as input a dataset of size \( n = 1,000,000 \) (see `feature.npy` and `obs.npy`) and returns a parameter vector \( \theta \in \mathbb{R}^{10} \), since \( d = 10 \) in this problem. The parameter vector returned by the code is expected to be "close" to the true parameter vector \( \theta^* \) that generated the observations \( Y_i \).

### Project Outcomes

The project successfully achieved its objectives, with the implemented logistic regression model returning an optimized parameter vector \( \theta \) that closely approximates the true parameter vector \( \theta^* \). The model was able to:

1. **Run Efficiently:** The ML code executes in under 5 seconds on a Google Colab notebook.
2. **Achieve High Accuracy:** The returned parameter vector \( \theta \) closely approximates the true parameter vector \( \theta^* \), minimizing the training loss on the dataset.

### Solution Approach

The minimization of \( E[Y_i | X_i] \) using logistic regression was implemented using the `minimize` function from the `scipy.optimize` library, a well-established method in the field of ML. This function was chosen for its efficiency and robustness, outperforming basic gradient descent methods. 

#### Objective Function

The objective function implemented is the Log Loss function, which is standard in binary classification tasks within ML. Additionally, the Jacobian (gradient) of the Log Loss function was implemented to facilitate the optimization process.

#### Optimization Method

For the minimization process, the `'L-BFGS-B'` method was chosen. This is a variant of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm, optimized for handling large datasets typical in ML projects. This method was selected to balance computational efficiency and accuracy.

### Accuracy Testing

The accuracy of the logistic regression model was tested by applying the optimized \( \theta \) to the model \( E[Y_i | X_i] \), using the `features.npy` dataset. Predictions were then compared against the actual observations in `obs.npy`. The accuracy was calculated based on binary classification, yielding satisfactory results that meet the project requirements.

## Installation

To run this project, ensure you have Python installed along with the following ML-related libraries:

```bash
pip install numpy scipy
```

## Usage Guide

To use this project, follow these steps:

1. **Open the Jupyter Notebook:**
   - In your terminal or command prompt, navigate to the root directory of this project where the `logistic_regressor.ipynb` file is located.
   - Start Jupyter Notebook by running:
     ```bash
     jupyter notebook
     ```
   - In the Jupyter interface, open the `logistic_regressor.ipynb` file.

2. **Load the Data:**
   - The notebook is set up to load the datasets `features.npy` and `obs.npy` automatically. Ensure these files are in the same directory as the notebook.

3. **Understand the Code:**
   - The notebook is organized into cells with explanations provided throughout. Start by reading the markdown cells for a clear understanding of each step.
   - The core logistic regression model is implemented in the notebook, utilizing the `scipy.optimize.minimize` function for parameter optimization.

4. **Run the Notebook:**
   - Execute each cell sequentially to perform the logistic regression analysis.
   - The notebook will:
     - Load the dataset.
     - Define the logistic regression model and the associated loss function.
     - Optimize the model parameters using the `L-BFGS-B` method.
     - Output the optimized parameter vector \( \theta \) and the model's accuracy.

5. **Review the Results:**
   - After running all cells, review the results at the end of the notebook. The optimized parameter vector \( \theta \) and the accuracy of the model on the provided data will be displayed.

6. **Experiment:**
   - Feel free to modify the code to test different configurations, such as changing the optimization method or adjusting the dataset size. This is a great way to deepen your understanding of logistic regression and ML model optimization.

## Contribution

Contributions to improve the ML model or add new features are welcome! Please fork the repository, create a feature branch, and submit a pull request for review.
