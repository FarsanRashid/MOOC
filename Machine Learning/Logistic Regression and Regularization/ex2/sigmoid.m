function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
for row_index = 1:size(z,1)
    for col_index = 1: size(z,2)
    val = z(row_index, col_index)
    g(row_index, col_index) = 1/(1 + exp(-val))
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
end
% =============================================================

end
