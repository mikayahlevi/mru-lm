# mru-lm

## How to Run

Use the command `python main.py --device=cuda --dataset=tiny_stories.py`. Set `--dataset=shakespeare_char` for the character-level Shakespeare dataset.

## What is this?

### Introduction

This is a project which implements the matrix recurrent unit (MRU) in place of attention. This repo is forked from my repo transformer-train-script.
I have limited compute and experience with datascience, so I haven't been able to test the LM on much other than a toy dataset, shakespeare-char. Based on the testing on that dataset, the MRU seems to work well in comparison to attention. The MRU also has the advantage of having linear complexity in regards to sequence length, in contrast to attention's quadratic complexity.
![MRU-LM vs Transformer](mru-lm_vs_transformer.png)

The above loss plot is the first train attempt, using the independent-heads branch of this repo and my other repo <https://github.com/mikayahlevi/transformer-train-script>.

### Moving Forward

Firstly, I would like to test this on larger and more informative datasets. If anyone wants to help me with this, reach out to me at <mikayahlevi@gmail.com> or through GitHub (if that's a thing, I'm not experienced with GitHub lol). Secondly, the MRU is still relatively slow compared to the amount of theoretical operations it should take, so I would like to investigate writing a CUDA kernel or just trying to optimize the PyTorch code.

### Explanation

#### General Idea

The idea of a matrix recurrent unit is dictated by the update rule $H_t = H_{t-1} X_{t-1}$,  and $H_1 = X_1$ where $X$ and $H$ are $\mathbb{R}^{s \times n \times n}$ sequences of square matrices. I tried generating matrix $X$ by different methods in the different branches. All of the ways to generate X and the output, Y, are arbitrary combinations of linears and reshapes and just based on what I found worked well.
Matrix multiplication is associative but not commutative. The associativity means I can compute the cumulative matrix product using an (inclusive) parallel scan. The lack of commutativity means that the order of tokens is automatically incorporated into the MRU.

#### Details of the Computation

The efficient version of the computation may have been figured out elsewhere, but I couldn't find any other sources that do this, so I will show the derivation here.
Sorry for my most likely incorrect mathematical notation. I am not well versed in the math fields that this computation involves. Note that the $^T$ symbol refers to transposing the last two dimensions.
The closed form ($1 \leq j \leq s$) for the MRU is

$$
H_j = \prod_{i=1}^{j} X_i
$$

The forward pass for the parallel scan is reasonably simple. I just used the Hillis-Steele prefix sum algorithm (except matmul instead of addition). The current implementation with the Hillis-Steele algorithm is somewhat inefficient for low parallelism because it does $O(n log_2 n)$ operations if $n$ is the sequence length, compared to another algorithm I plan on adding a branch for, the Brent-Kung algorithm, which does $O(n)$ operations but conversely has half the parallelism.

The backwards pass for the MRU way more complicated. $\frac{\partial F(H_j)}{H_j}$ represents output gradient of $H$, or the the partial derivative in respect to the rest of the network and the loss function. The closed form for the partial derivative is

$$
\frac{\partial F(H_j)}{\partial X_i} =
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i = 1 \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=2}^{j} X_k \right)^T & \text{if } j > i = 1 \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \neq 1 \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} X_k \right)^T & \text{if } j > i \neq 1 \\
0 & \text{if } j < i
\end{cases}
$$

The gradient of $X_i$ is

$$
\nabla X_i = \sum_{j=1}^{s} \frac{\partial F(H_j)}{\partial X_i}
$$

The expanded gradient of $X_j$ is

$$
\nabla X_i = \sum_{j=1}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i = 1 \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=2}^{j} X_k \right)^T & \text{if } j > i = 1 \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \neq 1 \\
\left(\prod_{k=1}^{i-1} X_k \right)^T \frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} X_k \right)^T & \text{if } j > i \neq 1 \\
0 & \text{if } j < i
\end{cases}
$$

If we define $A_{i+1} = H_{i}^T$ and $A_1 = I$, by factoring out $A_i$ the expression can be rewritten like:

$$
\nabla X_i = A_i \sum_{j=1}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} X_k \right)^T & \text{if } j > i \\
0 & \text{if } j < i
\end{cases} = A_i \sum_{j=i}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} X_k \right)^T & \text{if } j > i
\end{cases}
$$

I'll call the second part of the gradient a new variable, $B_i$:

$$
B_i = \sum_{j=i}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} X_k \right)^T & \text{if } j > i
\end{cases}
$$

You can see $B_s = \frac{\partial F(H_s)}{\partial H_s}$. The recurrent form for $B$ is $B_i = \frac{\partial F(H_i)}{\partial H_i} + B_{i+1} X_{i+1}^T$. $B_i$ can also be found with this expression:

$$
\begin{bmatrix}
0 & 0 \\
B_i & I
\end{bmatrix} =
\begin{bmatrix}
0 & 0 \\
B_{i+1} & I
\end{bmatrix}
\begin{bmatrix}
X_{i+1}^T & 0 \\
\frac{\partial F(H_i)}{\partial H_i} & I
\end{bmatrix} =
\begin{bmatrix}
0 & 0 \\
B_{s} & I
\end{bmatrix}
\begin{bmatrix}
X_{s}^T & 0 \\
\frac{\partial F(H_{s-1})}{\partial H_{s-1}} & I
\end{bmatrix}
\begin{bmatrix}
X_{s-1}^T & 0 \\
\frac{\partial F(H_{s-2})}{\partial H_{s-2}} & I
\end{bmatrix} \ldots
\begin{bmatrix}
X_{i+1}^T & 0 \\
\frac{\partial F(H_{i+1})}{\partial H_{i+1}} & I
\end{bmatrix}
\begin{bmatrix}
X_{i+1}^T & 0 \\
\frac{\partial F(H_{i})}{\partial H_{i}} & I
\end{bmatrix}
$$

If we let

$$
U_i = \begin{cases} X_{i+1}^T & \text{if } i \neq s \\
0 & \text{if } i = s \end{cases}
$$

and

$$
L_i = \frac{\partial F(H_i)}{\partial H_i}
$$

then we can express the equation with $B_i$ like:

$$
\begin{bmatrix}
0 & 0 \\
B_i & I
\end{bmatrix} =
\prod_{k=0}^{s-i}
\begin{bmatrix}
U_{s-k} & 0 \\
L_{s-k} & 1
\end{bmatrix}
$$

Which can be computed with a reverse parallel scan because matrix multiplication is associative.

Combining all of this, we get the final gradient for the input matrices, $X$, which is $\nabla X_i = A_i B_i$, which can be efficiently computed using a parallel scan and the output of the forward pass.
