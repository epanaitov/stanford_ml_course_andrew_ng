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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


reg_sum = 0.0;
for j = 1:size(theta)

	g_sum = 0.0;
	j_sum = 0.0;
	for i = 1:m
		
		z = 0.0;
		for t = 1:size(theta)
			z = z + theta(t)*X(i, t);
		end

		h = sigmoid(z);

		g_sum = g_sum + (h - y(i))*X(i, j);
	
		j_sum = j_sum - y(i)*log(h) - (1 - y(i))*log(1 - h);

	end
	
	grad(j) = g_sum/m;
	
	if j > 1
		grad(j) = grad(j) + lambda*theta(j)/m;
		reg_sum = reg_sum + theta(j)^2;
	end
	

end

J = j_sum/m + lambda*reg_sum/(2*m);


% =============================================================

end
