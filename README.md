# Curve fit GA-Initialized
Curve fit exploration capacity increased through a simple genetic algorithm. This algorithm iteratively initiates the curve-fit function with different initial values, checking the quality of the results and heuristically creating a number of new initial values to try to improve the model fitting.

This algorithm stops after the configured generations (iterations) are reached. A error-threshold stop criteria, or similar, should be implemented to avoid unnecessary computing after reaching enough quality in the solutions.
