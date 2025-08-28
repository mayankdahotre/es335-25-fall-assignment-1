# ES 335 Assignment 1 - Task 0 Question 4: Time Complexity Analysis of Decision Trees

## Overview
This task involves a comprehensive empirical analysis of the time complexity of decision tree algorithms. We systematically vary dataset parameters (number of samples N and number of features M) to validate theoretical time complexity bounds for both training and prediction phases across different input-output type combinations.

## Theoretical Time Complexity

### Training Phase
- **Time Complexity**: O(N × M × 2^d)
  - N: Number of training samples
  - M: Number of features
  - d: Maximum depth of the tree
- **Explanation**: For each node, we evaluate all M features across all N samples, and in the worst case, we create 2^d nodes

### Prediction Phase
- **Time Complexity**: O(N × d)
  - N: Number of test samples
  - d: Maximum depth of the tree
- **Explanation**: Each sample traverses at most d nodes in the tree

## Experimental Setup

### Dataset Configurations
We test four combinations of input and output types:
1. **Real Input, Real Output**: Continuous features, continuous target
2. **Real Input, Discrete Output**: Continuous features, categorical target
3. **Discrete Input, Real Output**: Categorical features, continuous target
4. **Discrete Input, Discrete Output**: Categorical features, categorical target

### Parameter Variations
- **N (Sample Size)**: Varied from 10 to 200 (keeping M=5 constant)
- **M (Feature Count)**: Varied from 2 to 20 (keeping N=50 constant)
- **Maximum Depth**: Fixed at 5 for all experiments
- **Repetitions**: Each experiment repeated 5 times for statistical reliability

## Results and Analysis

### Training Time Complexity

#### Linear Relationship with N
The plots below demonstrate the linear relationship between training time and number of samples:

![Real Input Real Output - Training vs N](Asst0_TC_plots/real_input_real_output%20wrt%20N%20Training.png)
![Real Input Real Output - Training vs M](Asst0_TC_plots/real_input_real_output%20wrt%20M%20Training.png)

**Key Observations:**
- Training time increases linearly with N (first plot)
- Training time increases linearly with M (second plot)
- Both trends confirm the O(N × M × 2^d) theoretical complexity

#### Variations Across Input-Output Types
Different input-output combinations show similar linear trends but with varying slopes:

**Real Input, Real Output:**
- Smoothest linear progression
- Consistent performance across parameter ranges

**Real Input, Discrete Output:**
- Similar to real-real case
- Slightly faster due to simpler output processing

**Discrete Input, Real Output:**
- Comparable performance to continuous inputs
- One-hot encoding overhead minimal

**Discrete Input, Discrete Output:**
- Fastest overall performance
- Both input and output processing optimized for categorical data

### Testing Time Complexity

#### Linear Relationship with N, Constant with M
The prediction phase shows the expected O(N × d) complexity:

![Real Input Real Output - Testing vs N](Asst0_TC_plots/real_input_real_output%20wrt%20N%20Testing.png)
![Real Input Real Output - Testing vs M](Asst0_TC_plots/real_input_real_output%20wrt%20M%20Testing.png)

**Key Observations:**
- Testing time increases linearly with N (first plot)
- Testing time remains constant with M (second plot)
- Confirms O(N × d) theoretical complexity

## Detailed Results by Category

### Real Input Real Output
![Training vs N](Asst0_TC_plots/real_input_real_output%20wrt%20N%20Training.png)
![Training vs M](Asst0_TC_plots/real_input_real_output%20wrt%20M%20Training.png)
![Testing vs N](Asst0_TC_plots/real_input_real_output%20wrt%20N%20Testing.png)
![Testing vs M](Asst0_TC_plots/real_input_real_output%20wrt%20M%20Testing.png)

### Real Input Discrete Output
![Training vs N](Asst0_TC_plots/real_input_discrete_output%20wrt%20N%20Training.png)
![Training vs M](Asst0_TC_plots/real_input_discrete_output%20wrt%20M%20Training.png)
![Testing vs N](Asst0_TC_plots/real_input_discrete_output%20wrt%20N%20Testing.png)
![Testing vs M](Asst0_TC_plots/real_input_discrete_output%20wrt%20M%20Testing.png)

### Discrete Input Real Output
![Training vs N](Asst0_TC_plots/discrete_input_real_output%20wrt%20N%20Training.png)
![Training vs M](Asst0_TC_plots/discrete_input_real_output%20wrt%20M%20Training.png)
![Testing vs N](Asst0_TC_plots/discrete_input_real_output%20wrt%20N%20Testing.png)
![Testing vs M](Asst0_TC_plots/discrete_input_real_output%20wrt%20M%20Testing.png)

### Discrete Input Discrete Output
![Training vs N](Asst0_TC_plots/discrete_input_discrete_output%20wrt%20N%20Training.png)
![Training vs M](Asst0_TC_plots/discrete_input_discrete_output%20wrt%20M%20Training.png)
![Testing vs N](Asst0_TC_plots/discrete_input_discrete_output%20wrt%20N%20Testing.png)
![Testing vs M](Asst0_TC_plots/discrete_input_discrete_output%20wrt%20M%20Testing.png)

## Sources of Variation

### Expected Variations
1. **Tree Structure Variability**: Actual number of nodes may be less than 2^d
2. **System Performance**: Background processes and CPU throttling
3. **Memory Access Patterns**: Cache performance varies with data size
4. **Implementation Overhead**: Python interpreter and library call overhead

### Statistical Considerations
- Error bars represent ±1σ standard deviation across 5 repetitions
- Some plots show non-perfect linear fits due to system variability
- Overall trends consistently match theoretical expectations

## Performance Insights

### Training Phase Efficiency
- **Best Performance**: Discrete input, discrete output
- **Worst Performance**: Real input, real output
- **Difference**: Approximately 20-30% performance gap
- **Reason**: Categorical data processing is more efficient than continuous splitting

### Prediction Phase Efficiency
- **Consistent Performance**: All input-output combinations show similar prediction times
- **Scalability**: Linear scaling with sample size across all configurations
- **Independence**: Prediction time independent of feature count (as expected)

## Conclusion

The empirical analysis strongly validates the theoretical time complexity of decision trees:

1. **Training Complexity Confirmed**: O(N × M × 2^d) relationship clearly demonstrated
2. **Prediction Complexity Confirmed**: O(N × d) relationship consistently observed
3. **Implementation Correctness**: Our algorithm follows expected computational patterns
4. **Scalability Understanding**: Clear insights into performance characteristics for different data types

This analysis provides confidence in our decision tree implementation and offers practical guidance for choosing appropriate algorithms based on dataset characteristics and computational constraints.
