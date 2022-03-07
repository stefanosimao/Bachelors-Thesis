# Bachelors-Thesis

## Abstract
Machine Learning and Parallel Programming are a significant part of modern Computer Science. The former with
applications such as image recognition, medical diagnosis, self-driving cars and many others. The latter is a con-
sequence of trying to compensate for the fact that Moore’s Law is not valid anymore and the constant increase in
the usage of multicore processors. The use of increasingly fast hardware can provide an enormous impact on the
performance of the training algorithms by adapting the algorithms to exploit parallelism and execute faster. The goal
of this project is to investigate existing Parallel Programming strategies to distribute the work of Machine Learning al-
gorithms for training Deep Neural Networks and propose a novel algorithm that reduces communication complexity.
This algorithm is a hybrid parallel stochastic gradient descent (SGD) method that significantly reduces the commu-
nication between the nodes while improving the generalization performance of the standard parallel SGD method.
This hybrid approach exploits both NVIDIA Collective Communication Library (NCCL) and Graphics Processing Units
(GPUs), to speed up the evaluation of the loss function and its derivatives.

