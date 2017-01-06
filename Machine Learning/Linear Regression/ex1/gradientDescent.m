function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta_zero = theta(1,1)
    prediced_values = X * theta
    difference = prediced_values - y
    col = X(:,1)
    
    multi = col(:,1).* difference(:,1); 
    total = sum(multi)
    theta_zero = theta_zero -( (alpha/m) * total )
    
    
    theta_one = theta(2,1)
    prediced_values_1 = X * theta
    difference_1 = prediced_values - y
    col_1 = X(:,2)
    
    multi_1 = col_1(:,1).* difference_1(:,1); 
    total_1 = sum(multi_1)
    theta_one = theta_one -( (alpha/m) * total_1 )

    theta(1,1) = theta_zero
    theta(2,1) = theta_one





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
