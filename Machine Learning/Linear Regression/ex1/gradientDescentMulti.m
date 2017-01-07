function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    theta_i = theta;
    for thete_index = 1:length(theta)
        theta_i(thete_index) = theta(thete_index);
        prediced_values = X * theta;
        difference = prediced_values - y;
        col = X(:,thete_index);
        multi = col(:,1).* difference(:,1); 
        total = sum(multi);
        theta_i(thete_index) = theta_i(thete_index) -( (alpha/m) * total );
        
    end
    theta = theta_i;







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
