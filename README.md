# Applied ML: Decision Trees, Neural Networks, and Behavioral Cloning

A comprehensive project exploring decision trees, manually implemented neural networks, and behavioral cloning using Keras through practical implementations and analysis.

## Problem Descriptions

### Problem 1: Decision Trees
- **Task**: Train a decision tree model to recommend seeds for sowing based on soil data.
- **Dataset**: `crop.csv`
- **Details**:
  - Clean and preprocess the data as needed.
  - Use built-in classifiers to train the decision tree model.

### Problem 2: Neural Networks
- **Task**: Train a neural network model to recommend seeds for sowing based on soil data.
- **Dataset**: `crop.csv`
- **Details**:
  - Implement the neural network manually using matrix operations, without using Keras.
  - Clean and preprocess the data as needed.

### Problem 3: Behavioral Cloning
- **Task**: Train a neural network using Keras to solve the MiniGrid-Empty-Random-6x6-v0 environment.
- **Details**:
  - Use `record.py` to save training data by manually solving the environment.
  - Implement the training process in `training.py`, using the recorded data.
  - Implement the model loading and testing in `test.py`, ensuring the model achieves an average reward greater than 0.6 over 10 consecutive tests.

## Running the Project
1. Ensure you have the required dependencies installed as specified in `requirements.txt`.
2. Run the provided Python scripts for each task in the respective order.
3. Review the output and comments for insights and analysis.

## Dependencies
- Python 3.x
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `keras`, `tensorflow`, `gym`, `minigrid`

## Notes
- Each task's implementation is self-contained within its respective script.
- Comments and explanations are provided within the scripts for better understanding.
- Data recording must be performed locally due to window display requirements. Training can be executed on Colab, but testing requires local execution for window display or setting the LOCAL flag to False for Colab testing.
