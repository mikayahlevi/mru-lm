# mru-lm

## What is this?

### Introduction

This is a project which implements the matrix recurrent unit (MRU) in place of attention. This repo is forked from my repo transformer-train-script.

### Explanation

#### General Idea

The idea of a matrix recurrent unit is that dictated by the update rule $H_t = H_{t-1} X_{t-1}$,  and $H_1 = X_1$ where $X$ and $H$ are matrices. I tried generating matrix $X$ by different methods in the different branches. All of the ways to generate X and the output, Y, are arbitrary combinations of linears and reshapes and just based on what I found worked well.  
Matrix multiplication is associative but not commutative. The associativity means I can compute the cumulative matrix product using an (inclusive) parallel scan. The lack of commutativity means that the order of tokens is automatically incorporated into the MRU.

#### Details of the Parallel Scan

Sorry for my most likely incorrect mathmatical notation. I am not well versed in the math fields that this scan involves. Note that the $^T$ symbol refers to transposing the last two dimensions.
The closed notation ($s$ is the sequence length) for the MRU is 

$$
H = \prod_{i=1}^{s} X_i
$$

The forward pass for the parallel scan is reasonably simple. I just used the Hillis-Steele prefix sum algorithm (except matmul instead of addition). The current implementation with the Hillis-Steele algorithm is somewhat inefficient for low parallelism because it does $O(n log_2 n)$ operations if $n$ is the sequence length, compared to another algorithm I plan on adding a branch for, the Brent-Kung algorithm, which does $O(n)$ operations but conversely has half the parallelism.

The backwards pass for the MRU way more complicated. $\partial F(H_j)$ represents output gradient of H, or the the derivative in respect to the rest of the network and the loss function. The closed notation for the partial derivative is

$$
\frac{\partial F(H_j)}{\partial X_i} = 
\begin{cases} 
\frac{\partial F(H_j)}{\partial H} & \text{if } i = j = 1 \\
\frac{\partial F(H_j)}{\partial H} \left(\prod_{k=2}^{j} X_k \right)^T & \text{if } i = 1 \text{ and } j > 1 \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_j)}{\partial H} & \text{if } i = j = s \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_j)}{\partial H} \left(\prod_{k=i+1}^{j} X_k \right)^T & \text{if } 1 < i \leq j < s \\
0 & \text{if } i > j 
\end{cases}
$$

The gradient of $X_i$ is

$$
\nabla X_i = \sum_{l=i}^{s} \frac{\partial F(H_l)}{\partial X_i}
$$

The expanded gradient of $X_i$ is

$$
\nabla X_i = \sum_{l=i}^{s} 
\begin{cases} 
\frac{\partial F(H_l)}{\partial H} & \text{if } i = l = 1 \\
\frac{\partial F(H_l)}{\partial H} \left(\prod_{k=2}^{l} X_k \right)^T & \text{if } i = 1 \text{ and } l > 1 \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_l)}{\partial H} & \text{if } i = l = s \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_l)}{\partial H} \left(\prod_{k=i+1}^{l} X_k \right)^T & \text{if } 1 < i \leq l < s \\
0 & \text{if } i > l 
\end{cases}
$$

If we define $A_{i+1} = H_{i}^T$ and $A_1 = I$, by factoring out $ A_i$ the expression can be rewritten like:

$$
\nabla X_i = A_i \sum_{l=i}^{s} 
\begin{cases} 
\frac{\partial F(H_l)}{\partial H} & \text{if } i = l = 1 \\
\frac{\partial F(H_l)}{\partial H} \left(\prod_{k=2}^{l} X_k \right)^T & \text{if } i = 1 \text{ and } l > 1 \\
\frac{\partial F(H_l)}{\partial H} & \text{if } i = l = s \\
\frac{\partial F(H_l)}{\partial H} \left(\prod_{k=i+1}^{l} X_k \right)^T & \text{if } 1 < i \leq l < s \\
0 & \text{if } i > l 
\end{cases}
$$

## How to Use

Use the command `python main.py --device=cuda --dataset=tiny_stories.py`. Set `--dataset=shakespeare_char` for the character-level Shakespeare dataset.
