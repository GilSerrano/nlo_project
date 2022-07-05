# Nonlinear Optimization Project - A Distributed Augmented Lagrangian Method for Model Predictive Control

## Introduction

In this repository, you can find the Python source code for the project of Nonlinear Optimization (2021/2022).

## Running the code

There are three implementations: 1) distributed solver; 2) centralised solver; 3) centralised solver using a matrix formulation.

Each of the solvers is run from inside the ```scripts/``` directory, i.e., you need to ```cd``` into it before running the script.

### Distributed Solver

To run the distributed solver, run the following command:

```
python3 solver_distributed.py <problem_file.yaml>
```

where ```<problem_file.yaml>``` is one of the problem files in the ```config/``` directory. If you would like, for example, to solve the ```mpc_3agents_2d_simpleInteractions.yaml``` problem, the command would be:


```
python3 solver_distributed.py mpc_3agents_2d_simpleInteraction.yaml
```
### Centralised Solver

To run the centralised solver, run the following command:

```
python3 solver_centralised.py <problem_file.yaml>
```


### Centralised Solver with Matrix formulation

To run the centralised solver with matrix formulation, run the following command:

```
python3 solver_centralised_matrix.py <problem_file.yaml>
```
