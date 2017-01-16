function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

for index = 1:m
   y_i = y(index)
   x_i = X(index,:)
   h_theta_i = sigmoid(x_i * theta)
   J = J +  ( -y_i * log(h_theta_i) - (1-y_i) * log(1-h_theta_i ) )
end

J = J/m

total = sum(theta(index).^2)

J = J + (lambda/(2*m)) * 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
