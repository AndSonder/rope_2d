# rope_2d

English | [简体中文](README_CN.md)

## 1. Background

In NLP, the positional information of language is one-dimensional, meaning we need to inform the model about the position of each word in the sentence. However, in CV, the positional information of an image is two-dimensional, meaning we need to inform the model about the row and column positions of each feature. Here, two-dimensional means that complete positional information requires two numbers, not referring to the dimensionality of the position vector.

If we directly use the position encoding from the Transformer, it would cause the model to be unable to distinguish features at different positions. Therefore, two-dimensional positional encoding needs to be introduced.

## 2. Comparison of 1D and 2D ROPE

Assume the dimension of $ q $ is 4, the one-dimensional position encoding is as follows:

$$
f(q, m) = R_mq = \left(\begin{array}{cc:cc}
\cos m \theta & -\sin m \theta & 0 & 0 \\
\sin m \theta & \cos m \theta & 0 & 0 \\
\hdashline 0 & 0 & \cos m \theta & -\sin m \theta \\
0 & 0 & \sin m \theta & \cos m \theta
\end{array}\right) \left(\begin{array}{c} q_0 \\ q_1 \\ q_2 \\ q_3 \end{array}\right)
$$

The extended formula for multiple dimensions is as follows:

$$
\left(\begin{array}{ccccccc}
\cos m \theta_0 & -\sin m \theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_0 & \cos m \theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_1 & -\sin m \theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_1 & \cos m \theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2-1} & -\sin m \theta_{d / 2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2-1} & \cos m \theta_{d / 2-1}
\end{array}\right)\left(\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right)
$$

The above sparse matrix can be simplified as:

$$
\left(\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos m \theta_0 \\
\cos m \theta_0 \\
\cos m \theta_1 \\
\cos m \theta_1 \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{d / 2-1}
\end{array}\right)+\left(\begin{array}{c}
-q_1 \\
q_0 \\
-q_3 \\
q_2 \\
\vdots \\
-q_{d-1} \\
q_{d-2}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin m \theta_0 \\
\sin m \theta_0 \\
\sin m \theta_1 \\
\sin m \theta_1 \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{d / 2-1}
\end{array}\right)
$$

Two-dimensional positional encoding can be simply understood as calculating the one-dimensional positional encoding in two directions separately and then concatenating them together. For the case where $ q $ is 4-dimensional, it is as follows:

$$
f(q, x, y) = R_{xy}q = \left(\begin{array}{cc:cc}
\cos x \theta & -\sin x \theta & 0 & 0 \\
\sin x \theta & \cos x \theta & 0 & 0 \\
\hdashline 0 & 0 & \cos y \theta & -\sin y \theta \\
0 & 0 & \sin y \theta & \cos y \theta
\end{array}\right) \left(\begin{array}{c} q_{0,0} \\ q_{0,1} \\ q_{1,0} \\ q_{1,1} \end{array}\right)
$$

The extended formula for multiple dimensions is as follows:




Of course, this sparse matrix can also be simplified as:


$$
\left(\begin{array}{c}
q_{0,0} \\
q_{0,1} \\
q_{0,2} \\
q_{0,3} \\
\vdots \\
q_{d/4-1,d/4-2} \\
q_{d/4-1,d/4-1} \\
q_{d/4-1,d/4} \\
q_{d/4-1,d/4+1} \\
\vdots \\
q_{d/2-1,d/2-2} \\
q_{d/2-1,d/2-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos x \theta_0 \\
\cos x \theta_0 \\
\cos x \theta_1 \\
\cos x \theta_1 \\
\vdots \\
\cos x \theta_{d / 4-1} \\
\cos x \theta_{d / 4-1} \\
\cos y \theta_0 \\
\cos y \theta_0 \\
\vdots \\
\cos y \theta_{d / 4-1} \\
\cos y \theta_{d / 4-1}
\end{array}\right)+\left(\begin{array}{c}
-q_{0,1} \\
q_{0,0} \\
-q_{0,3} \\
q_{0,2} \\
\vdots \\
-q_{d/4-1,d/4-2} \\
q_{d/4-1,d/4-1} \\
-q_{d/4-1,d/4} \\
q_{d/4-1,d/4+1} \\
\vdots \\
-q_{d/2-1,d/2-2} \\
q_{d/2-1,d/2-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin x \theta_0 \\
\sin x \theta_0 \\
\sin x \theta_1 \\
\sin x \theta_1 \\
\vdots \\
\sin x \theta_{d / 4-1} \\
\sin x \theta_{d / 4-1} \\
\sin y \theta_0 \\
\sin y \theta_0 \\
\vdots \\
\sin y \theta_{d / 4-1} \\
\sin y \theta_{d / 4-1}
\end{array}\right)
$$

As can be seen, the calculation method of two-dimensional positional encoding is the same as that of one-dimensional, just computed separately in two directions and then concatenated together. From an implementation perspective, only the logic for calculating $\sin$ and $\cos$ needs to be modified.