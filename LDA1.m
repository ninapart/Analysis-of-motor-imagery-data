function [ W, b, y_pred ] = LDA1( X, y )
% LDA1 perform linear discriminant analysis
% INPUT:
% X - sample matrix where each row is an observation
% y - column vector with labels of +1 or -1
% OUTPUT:
% W, b - weight vector and bias for classification, such that: y = sign(X*W+b);
% y_pred - the results of the classification appliied to the data

ind1 = find(y==1);
ind2 = find(y==-1);
N1 = length(ind1);
N2 = length(ind2);
X1 = X(ind1,:);
X2 = X(ind2,:);
mue1_hat = mean(X1);
mue2_hat = mean(X2);
delta_X1 = X1 - ones(N1,1)*mue1_hat;
Sig1 = delta_X1'*delta_X1;
delta_X2 = X2 - ones(N2,1)*mue2_hat;
Sig2 = delta_X2'*delta_X2;

W = inv(Sig1 + Sig2)*(mue1_hat - mue2_hat)';
b = -0.5*dot(W,(mue1_hat + mue2_hat));
if nargout>2,
    y_pred = sign(X*W+b);
end

end

