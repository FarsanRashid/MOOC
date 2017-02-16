function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

one = ones(m,1);
X = [one X];

eye_matrix = eye(num_labels);
y = eye_matrix(y,:);

d2 = zeros(hidden_layer_size,1);
d3 = zeros(num_labels,1);

for i = 1:m
    x_i = transpose(X(i,:));
    total = 0;
    for k = 1:num_labels
        z_2 = Theta1 * x_i;
        a_2 = sigmoid(z_2);
        one = ones(1,1);
        a_2 = [one;a_2];
        z_3 = Theta2 * a_2;
        h_theta = sigmoid(z_3);
        total = total - y(i,k) * log(h_theta(k,1)) - (1- y(i,k))* log(1-h_theta(k,1));
        d3(k) = h_theta(k,1) - y(i,k);
    end
    z_2 = [one;z_2];
    d2 = (transpose(Theta2) * d3).*sigmoidGradient(z_2);
    d2 = d2(2:end);
    Theta1_grad = Theta1_grad + d2 * transpose(x_i);
    Theta2_grad = Theta2_grad + d3 * transpose(a_2);
    J = J+ total;
end
Theta1_grad = Theta1_grad ./m;
Theta2_grad = Theta2_grad ./m;

J = sum(J);
J= J/m;

C = Theta1.^2;
C = C(:,2:end);%We do not regularize theta_0

D = Theta2.^2;
D = D(:,2:end);


J = J + (lambda /(2*m) *( sum(C(:))+ sum(D(:))));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

for i = 1 : size(Theta1_grad,1)
    for j = 2 : size(Theta1_grad,2)
        Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m) * Theta1(i,j);
    end
end

for i = 1 : size(Theta2_grad,1)
    for j = 2 : size(Theta2_grad,2)
        Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m) * Theta2(i,j);
    end
end

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
