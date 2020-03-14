# CS420 - Machine Learning Projects

Repo of course assignments and projects in 2020 Spring.

***Here is some of my notes. Since GitHub doesn't support MathJax, the math formulas, unfortunately,  cannot be displayed correctly.***

## Linear Model

### Logistic Regression

#### Multi-class Classification

We want to maximize the likelihood $\Pi_{i = 1}^n p(y = y_i | x_i)$ where $n$ is the number of samples for training.

Take the logarithm, we need to minimize $L = -\sum_{i = 1}^n \log p(y = y_i | x_i)$. Let $L = \sum_{i = 1}^n L_i$, which means $L_i = -\log p(y = y_i | x_i)$.

By Softmax function, $p(y = y_i | x_i) = \frac{e^{W_{y_i}^T x_i}}{\sum_{j=1}^K e^{W_j^T x_i}}$. So $L_i = -W_{y_i}^Tx_i + \log\sum_{j = 1}^K e^{W_j^T x_i}$.

$$
\begin{aligned}
\frac{\partial L_i}{\partial W_j} &= [j = y_i](-x_i) + \frac{e^{W_j^T x_i}x_i}{\sum_{k = 1}^K e^{W_k^T} x_i} \\
&= [j = y_i](-x_i) + p(y_i = c_j | x_i) \cdot x_i
\end{aligned}
$$

So every sample contributes to $W_j$ for $j \in [K]$. If $j = y_i$, the gradient is $(p(y_i = c_j | x_i) - 1)x_i$. Otherwise, the gradient should be $p(y_i = c_j | x_i)x_i$.