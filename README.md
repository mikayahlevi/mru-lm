# mru-lm

## Introduction

This repo adds the option to use the matrix recurrent unit (MRU) in place of attention in a traditional GPT2-based transformer.

Run using main.py, which allows you to specify model, hyperparameter, and training configuration files or to set the parameters manually using CLI arguments. 

## Explanation of the MRU

### Inspiration

I came up with the MRU by asking what would happen if the weight matrix in a recurrent neural network was fully data-dependent. Exploring this, I dropped the additive terms, yielding: $H_t = H_{t-1} M_t$, $H_1 = M_1$. $M_t$ is the input state matrix and is a (selectable) function of $x_t$ that transforms the input embeddings into a matrix. $H_t$ is the  hidden state AKA output state matrix. The output embeddings, $y_t$, are a (selectable) function of $H_t$.

The following are some entailing properties: 

- Matrix multiplication is associative but not commutative. The associativity means I can compute the cumulative matrix product using a parallel scan. The lack of commutativity means that the MRU automatically incorporates the order of the tokens.
- The complexity of the computation scales with the cube of order/width of $M_t$. Therefore, we must choose the order of the input matrices $M_t$ to be significantly smaller than the weight matrix of an RNN, for example on the order of $\sqrt{d}$ ($d$ being model width).
- When processing the tokens sequentially, the network scales linearly with sequence length in contrast to attention which scales quadratically.

## Dimensions and Computational Complexity

For the rest of this document, let's call the sequence length $s$, the number of heads $h$, the embedding size of the network $d_e$, and the state size of the network $d_s$.
The head size, consequently, is $d_h = \frac{d_s}{h}$.
The matrix state order, or the width/height of the matrix states is $d_o = \sqrt{d_h} = \sqrt{\frac{d_s}{h}}$.
Lastly, the embedding state chunk size is $d_c = \frac{d_h}{d_o}$.

I've implemented parallel scans which require more raw computation but utilize DL hardware more efficiently. Recurrent processing can accomplish the operation in $s$ sequential steps, while the Hillis-Steele and Sklansky scan in $\log_2(s)$ steps, and the Brent-Kung scan in $2 \log_2(s)$ steps.
I repurposed the Brent-Kung and Hillis-Steele from prefix sums for cumulative matrix multiplication.


The most optimized and balanced mode for parallel computation is the Sklansky scan implemented with [a CUDA PyTorch extension](https://github.com/mikayahlevi/cuda_mru), but it is still in-progress and missing support for every input tensor shape. All of the computation modes are available in the mru_scans/ folder.

The number of operations for the MRU is:

- Using recurrence

$$
(s) (h) (d_o)^3 = (s) (d_h)^{-\frac{1}{2}} (d_s)^\frac{3}{2}
$$

- Using the Sklansky scan

$$
(\log_2 s) (\frac{s}{2}) (h)  (d_o)^3 = (\log_2 s) (\frac{s}{2}) (d_h)^{-\frac{1}{2}} (d_s)^\frac{3}{2}
$$

- Using the Brent-Kung scan

$$
2 (s) (h)  (d_o)^3 = 2 (s) (d_h)^{-\frac{1}{2}} (d_s)^\frac{3}{2}
$$

- Using the Hillis-Steele scan

$$
(\log_2 s) (s) (h)  (d_o)^3 = (\log_2 s) (s) (d_h)^{-\frac{1}{2}} (d_s)^\frac{3}{2}
$$

## Restructuring the Vectors into Matrices and Back

The MRU should take in a sequence of vectors and return a sequence of vectors, like any other traditional operation in a neural network. For now I'll ignore the batch and sequence dimensions and only focus on the final dimension. The input and output embeddings $x$ and $y$ are vectors, but the input and output hidden states $M$ and $H$ are matrices so we must have a way to convert vectors to matrices and matrices to vectors.

The simplest vector-matrix conversion method is reshaping and adding the identity (inverse vectorization and vectorization): 
$$
M = \text{reshape}(x W_{in}, h, d_o, d_o) + I
$$

$W_{in}$ is a $d_e \times d_s$ tensor.

$$
y = \text{reshape}(H W_{out}, d_e)
$$

Therefore, $W_{out}$ is a $h \times d_h \times d_c$ tensor, which has the result of essentially matrix-multiplying each head of $H$ by a unique weight matrix.

However, the determinant of $M$ is completely free to vary in this method. If the determinant tends lower or higher than 1, the hidden matrices can collapse or explode. In the [create_state_matrix.ipynb notebook](https://github.com/mikayahlevi/mru-lm/blob/main/create_state_matrix.ipynb), I explore stable variants, such as filling a skew-symmetric matrix with elements from a vector and taking the Cayley transform, guaranteeing an orthogonal matrix or creating LDU factors, allowing the determinants to be efficiently normalized.

## Comparison with Other Linear-Time Sequence Algorithms

Many recent proposed linear-time architectures reformulate attention by linearizing it. The keys and values can then be expressed by a single matrix, which is additively updated through time. Architectures like Mamba 2, DeltaNet, RWKV, and more use variations of the linear attention/state matrix duality.
This project drastically differs from the other linear-time algorithms because the state matrix is much smaller and the updates are multiplicative in contrast to additive. Some of the other projects multiply by a structured matrix each timestep, but the MRU uses full, dense matrices, focusing on maximizing the computation from one position in the sequence in another, at the cost of the information density of the states.

## Efficient Backward Pass Implementation

For the MRU, I've derived an efficient algorithm using a parallel scan to compute it.

An operator must be associative through time in order for it to be computed with a parallel scan.

The associativity of the forward pass is immediately apparent when viewing the closed form ($1 \leq j \leq s$):

$$
H_j = \prod_{i=1}^{j} M_i
$$

The backwards pass, on the other hand, is not associative at first glance. Here, I'll show how to reformulate it associatively.

$\frac{\partial F(H_j)}{\partial H_j}$ represents output gradient of $H$, or the the partial derivative in respect to the rest of the network and the loss function. The closed form for the partial derivative is

$$
\frac{\partial F(H_j)}{\partial M_i} =
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i = 1 \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=2}^{j} M_k \right)^T & \text{if } j > i = 1 \\
\left(\prod_{k=1}^{i-1} M_k \right)^T \frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \neq 1 \\
\left(\prod_{k=1}^{i-1} M_k \right)^T \frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} M_k \right)^T & \text{if } j > i \neq 1 \\
0 & \text{if } j < i
\end{cases}
$$

The gradient of $M_i$ is

$$
\nabla M_i = \sum_{j=1}^{s} \frac{\partial F(H_j)}{\partial M_i} =
\sum_{j=1}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i = 1 \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=2}^{j} M_k \right)^T & \text{if } j > i = 1 \\
\left(\prod_{k=1}^{i-1} M_k \right)^T \frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \neq 1 \\
\left(\prod_{k=1}^{i-1} M_k \right)^T \frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} M_k \right)^T & \text{if } j > i \neq 1 \\
0 & \text{if } j < i
\end{cases}
$$

If we define $A_{i+1} = H_{i}^T$ and $A_1 = I$, by factoring out $A_i$ the expression can be rewritten as:

$$
\nabla M_i = A_i \sum_{j=1}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} M_k \right)^T & \text{if } j > i \\
0 & \text{if } j < i
\end{cases} = A_i \sum_{j=i}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} M_k \right)^T & \text{if } j > i
\end{cases}
$$

I'll call the second part of the gradient a new variable, $B_i$:

$$
B_i = \sum_{j=i}^{s}
\begin{cases}
\frac{\partial F(H_j)}{\partial H_j} & \text{if } j = i \\
\frac{\partial F(H_j)}{\partial H_j} \left(\prod_{k=i+1}^{j} M_k \right)^T & \text{if } j > i
\end{cases}
$$

You can see $B_s = \frac{\partial F(H_s)}{\partial H_s}$. The recurrent form for $B$ is $B_i = \frac{\partial F(H_i)}{\partial H_i} + B_{i+1} M_{i+1}^T$. We can reformulate $B_i$ in terms of associative block matrix multiplications:

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
M_{i+1}^T & 0 \\
\frac{\partial F(H_i)}{\partial H_i} & I
\end{bmatrix} =
\begin{bmatrix}
0 & 0 \\
B_{s} & I
\end{bmatrix}
\begin{bmatrix}
M_{s}^T & 0 \\
\frac{\partial F(H_{s-1})}{\partial H_{s-1}} & I
\end{bmatrix}
\begin{bmatrix}
M_{s-1}^T & 0 \\
\frac{\partial F(H_{s-2})}{\partial H_{s-2}} & I
\end{bmatrix} \ldots
\begin{bmatrix}
M_{i+1}^T & 0 \\
\frac{\partial F(H_{i+1})}{\partial H_{i+1}} & I
\end{bmatrix}
\begin{bmatrix}
M_{i+1}^T & 0 \\
\frac{\partial F(H_{i})}{\partial H_{i}} & I
\end{bmatrix}
$$

If we let

$$
U_i = \begin{cases} M_{i+1}^T & \text{if } i \neq s \\
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
L_{s-k} & I
\end{bmatrix}
$$

Combining all of this, we get the final associatively computable gradient for the input state matrices, $M$, $\nabla M_i = A_i B_i$.
