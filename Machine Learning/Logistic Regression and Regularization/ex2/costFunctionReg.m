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

total = sum(theta.^2)

total = total - theta(1)^2 %We do not regularize theta_zero

J = J + (lambda/(2*m)) * total

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


total = 0
for index = 1:m
   y_i = y(index)
   x_i = X(index,:)
   h_theta_i = sigmoid(x_i * theta)
   total = total +  (h_theta_i - y_i) * X(index,1)
end
grad(1) = total / m

for theta_index = 2:size(theta)
    total = 0
    for index = 1:m
        y_i = y(index)
        x_i = X(index,:)
        h_theta_i = sigmoid(x_i * theta)
        total = total +  (h_theta_i - y_i) * X(index,theta_index)
    end
    total = total/m
    grad(theta_index) = total + (lambda/m) * theta(theta_index)
end

% =============================================================

end
