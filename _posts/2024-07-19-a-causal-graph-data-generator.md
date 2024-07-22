---
title: 'A Causal Graph Data Generator'
date: 2024-07-19
permalink: /posts/2024/07/a-causal-graph-data-generator/
tags:
  - causal reasoning
  - causal data generator
---

# A Causal Graph Data Generator
About a year ago I started my journey in causal reasoning. Initially, like many others, I struggled to fully grasp its intricacies. 

For this reason, and because like any aspiring academic I required toy datasets for benchmarking, I decided to code up my own synthetic causal data generator. The purpose of this first post is mainly to document how to use it.

Note this blog post does not intend to be an introduction to causal reasoning. If you are looking for this, I strongly recommend:

- [Causal Reasoning: Fundamentals and Machine Learning Applications](https://causalinference.gitlab.io/): A fantastic resource offering a rigorous deep dive into causal graphs, structural equations, do-calculus, identification of causal estimands, and more.

- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html): A light-hearted booklet designed for practitioners.

## Structural Causal Models (SCMs)
Let's dive straight into it! Here, I briefly provide some background to facilitate our discussion. In brief, SCMs comprise two key elements: a causal graph and structural equations.

### Causal Graphs

A structural causal model (SCM) is defined by its causal graph, that is a directed acyclic graph, or DAG, \\(G=(V, E)\\), and a joint probability distribution \\(p(\boldsymbol{x})\\) over a random vector \\(\boldsymbol{x} = (x_1, \ldots, x_n)\\). Each node \\(i \in V = \{1, \ldots, n\}\\) corresponds to a random variable \\(x_i\\), while every edge \\((i, j) \in E\\) signifies a direct causal link from variable \\(x_i \to x_j\\).

$$
G = (V, E)
$$

$$
p(\boldsymbol{x}) = p(x_1, \ldots, x_n)
$$

In an SCM, the causal relationships between variables are represented through these directed edges, indicating the direction of causality. This framework allows for the understanding and analysis of how changes in one variable affect others within the system.

### Creating DAGs with `RandomCausalGraphs`
`RandomCausalGraphs` supports two types of random DAGs - **Erdos-Renyi (ER)** and **Barabasi-Albert (SF)** graphs. Under the hood the library relies on `networkx` to generate the graphs themselves, but we ensure acyclicity by orienting the edges from lower-numbered to higher-numbered nodes. Also, we explicitly check if the generated graph is in fact a DAG and is acyclic. All you need to do is provide the number of nodes and edge probability and demonstrated in the `python` snippet below on this page.

![Causal Graph Visualization](/images/random_dag.png)

Above is an example Erdos-Renyi (ER) random DAG with 20 nodes generated with `RandomCausalGraphs`.  

### Structural Equations
The joint distribution \\(p(\boldsymbol{x})\\) admits the following factorisation:

$$
p(x_1, \ldots, x_n) = \prod_{j=1}^n p_j (x_j \mid \boldsymbol{x}_{pa(x_j)})
$$

Here, \\(pa(x_j)\\) indicates the parent set of node \\(j\\) within \\(G\\), with \\(\boldsymbol{x}_{pa(j)}\\) forming a vector encapsulating the parents' values.

We use structural equation models (SEMs) to represent the conditional distribution of a node \\(x_j\\) as a function of its parents, following the general form:

$$
p(x_j \mid \boldsymbol{x}_{pa(j)}) = f(\boldsymbol{x}_{pa(j)}, \epsilon_j)
$$

where \\(\epsilon_j\\) represents a noise variable.

### SEMs in `RandomCausalGraphs`

`RandomCausalGraphs` supports both additive: \\(x_j = f(\boldsymbol{x}_{pa(j)}) + \epsilon_j\\),

and non-additive: \\(x_j = f(\boldsymbol{x}_{pa(j)}, \epsilon_j)\\) noise models.

Furthermore, we support linear, non-linear and discrete transformations for \\( f(\cdot) \\).


- **Linear SEMs**: A linear model with additive noise. The noise variable can be sampled from Gaussian ('gauss'), exponential ('exp'), Gumbel ('gumbel'), uniform ('uniform') distributions.
- **Non-linear SEMs**: multi-layer perceptron or multiple interaction model. Both have additive noise ('mlp' or 'mim') or non-additive noise versions ('mlp-non-add' or 'mim-non-add') respectively.
- **Discrete and other models**: 'logistic', 'poisson'
- **Gaussian processes**: 'gp' or 'gp-add' depending on whether parent nodes are modelled jointly in a multi-dimensional GP or independently by applying a GP to each parent node and then summing the result.

An example how to create a CausalGraph object:

```python
import numpy as np
from src.synthetic_causal_graphs import CausalGraph
import matplotlib.pyplot as plt

# Define parameters for CausalGraph
n_nodes = 20  # number of nodes
p = 0.2  # edge probability
seed = 42
graph_type = 'ER'
sem_type = 'mlp'  # Multi-layer perceptron with additive noise
w_ranges = ((-2.0, -0.5), (0.5, 2.0))  # Disjoint weight ranges for generating node transformation weights. 

# Depending on number of nodes and edge probability, the generated random DAG might comprise several disconnected components.
# This is generally not desirable so the CausalGraph class checks if the DAG is weakly connected. 
max_attempts = 10  # Set the maximum number of attempts to create a weakly connected graph

# Try to create a weakly connected CausalGraph
for attempt in range(max_attempts):
    try:
        # Generate the CausalGraph
        model = CausalGraph(n_nodes, p, graph_type, sem_type, w_ranges, seed=seed)
        break  # Exit the loop if successful
    except AssertionError as e:
        print(f"Attempt {attempt + 1} failed: {e}")
else:
    raise RuntimeError(f"Failed to create a weakly connected CausalGraph after {max_attempts} attempts")
```

## Generating data

### Observational samples
We can sample observational data from the generated SCM by simply doing:

```python
noise_scale = 1.0
n_samples = 1000

# Simulate SEM to generate samples
X = model.simulate_sem(n_samples, noise_scale=noise_scale)  # Output shape: (n_samples, n_nodes). The columns of X follow the topological order of the nodes, that is 0, 1, 2,...n_nodes-1.
```

### Performing do-interventions
A strength of the SCM framework over and above providing a data-generating model for the native, observational case, lies in its ability to modify the underlying model, thereby permitting generation of output under, e.g., a hard intervention on a specific variable \\(x_j'\\).  

This process entails replacing its conditional distribution \\( p(x_j' | \boldsymbol{x}_{pa(j)}) \\) with an alternate distribution, such as a delta function \\(\delta(x_j' = x)\\), enforcing \\(x_j'\\) to assume the fixed value \\(x\\). 

Note the value \\(x_j'\\) assumes is no longer dependent on its parents, meaning these edges have been deleted in the intervention DAG \\( G_{do(x_j' = x)} \\).


**`RandomCausalGraphs`** allows you to perform hard interventions on individual nodes like this:

```python
# Simulate data with an intervention on node 10, setting its value to 0
# Shape of X_intervened is (n_samples, n_nodes). The columns follow the topological order of the nodes, that is 0, 1, 2,...n_nodes-1.
intervened_value = 0
intervened_node = 10
X_intervened = model.simulate_sem(n_samples=n_samples, intervened_node=intervened_node, intervened_value=intervened_value, noise_scale=noise_scale)
```

### Computing the Counterfactual

What happens when we perform hard interventions on nodes? The key point is that the underlying causal graph changes after the intervention. This means the observational distribution \\( p(\boldsymbol{x}) \\) and the interventional distribution \\( p_{G_{do(x_j' = x)}}(\boldsymbol{x}) \\) are different because the graph's connectivity changes!

How does `RandomCausalGraphs` handle this? Simply put, `RandomCausalGraphs` modifies the causal graph into the intervention DAG before simulation and uses a controlled random seed generator. This ensures that the same random processes (such as noise generation and weight assignment) are used for both the observational and interventional datasets, allowing us to calculate exact counterfactuals.

Let's see what this means in practical terms. In the example above, we intervened on node 10 by setting its value to 0. Since causal effects do not propagate to ancestors, the values for node 3, for example, across the simulated samples must match between the observational samples \\( X \\) and the interventional samples \\( X_{\text{intervened}} \\).

On the other hand, the causal effect will propagate to the descendants of node 10 after intervention. The Random ER graph visualization figure above shows the causal graph for this example, and we find node 10 has two descendants: nodes 15 and 18. We choose to inspect the samples of node 15 between \\( X \\) and \\( X_{\text{intervened}} \\), and expect to see a difference.

![Observational vs Interventional](/images/obs_vs_int.png)

And indeed, the histogram of values for the ancestor node 3 matches perfectly between the observational \\( X \\) and interventional \\( X_{\text{intervened}} \\) datasets, while they are noticeably different for the descendant node.

### Computing the Average Causal Effect of an Intervention
To estimate the causal response under an intervention, we utilize a fitness function that calculates an outcome \\( Y \\) from the samples. This is done by computing the weighted mean of a subset of selected variables and adding Gaussian noise.

```python
# Calculate fitness
from src.utils import fitness

fitness_obs, _, _ = fitness(X, noise_std=0.1, proportion=0.1, seed=seed, strategy='last_few')  # Fitness Values Shape: (n_samples,)
fitness_int, _, _ = fitness(X_intervened, noise_std=0.1, proportion=0.1, seed=seed, strategy='last_few')
# Calculate the causal effect of the intervention
ATE = np.mean(fitness_int - fitness_obs)
```
Ensure you use the same seed when calculating the fitness of both the observational and interventional samples. This consistency guarantees that the subset of variables used in the calculations matches.

Since each sample in the interventional dataset is a counterfactual of a matched observational sample, we can directly calculate the causal effect using the formula:

$$
E[Y(\text{do}(\text{node } 10)=0) - Y_{\text{obs}}] = \frac{1}{n_{\text{samples}}} \sum_{k=1}^{n_{\text{samples}}} \left[ Y(\text{do}(\text{node } 10)=0, k) - Y_{\text{obs}}, k \right]
$$

In our case, this boils down to a simple mean.
