# Graph Neural Networks (GraphNN)

Implementation of a node encoder network in PyTorch. Supports problems involving node classification through usage of sparse adjacency matrices.

Each layer follows the form:

H<sup>(k)</sup> = &sigma;(H<sup>(k-1)</sup>W<sub>0</sub> + D<sup>-1</sup>A<sub>I</sub>H<sup>(k-1)</sup>W<sub>1</sub>)

where &sigma; is a nonlinear activation function such as ReLU, D<sup>-1</sup>A<sub>I</sub> is the normalized degree matrix, and W<sub>i</sub> are weight matrices represented as Linear layers.
