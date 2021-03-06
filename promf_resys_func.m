function [ P ] = promf_resys_func( Y,R,feat_num,lambda)
% matrix factorization for recommender systems
%
%   reference: Koren, Yehuda, Robert Bell, and Chris Volinsky.
%           "Matrix factorization techniques for recommender systems."
%           Computer 8 (2009): 30-37.
%
%   solve the following objective function by the gradient descent:
%
%   [X, Theta] =
%       argmin_{X,Theta} 1/2||R.*(Y - X'*Theta - Offset)||_F^2 +
%       lambda/2(||X||_F^2 + ||Theta||_F^2 + ||b||_F^2)
%
%   where X, Theta are the latent feature matrices
%
%   X:          item_num x feat_num
%   Theta:      user_num x feat_num
%   b           1 x user_num
%   Y(ratings): item_num x user_num
%   
%   P:          rating matrice
%

    [item_num,user_num] = size(R);    

    % mean normalization: for new users
%     Ymean = zeros(item_num, 1);
%     Ynorm = zeros(item_num,user_num);
%     for i = 1:item_num
%         idx = find(R(i, :) == 1);
%         Ymean(i) = mean(Y(i, idx));
%         Ynorm(i, idx) = Y(i, idx) - Ymean(i);
%     end
%     Y = Ynorm;

    % initialization 
    X = randn(item_num, feat_num);
    Theta = randn(user_num, feat_num);
    b = randn(1, user_num);
    init_val = [X(:); Theta(:); b(:)];

    maxiter = 100;

    options = optimset('GradObj', 'on', 'MaxIter', maxiter);
    tic;
    vec_obj = fmincg (@(x)(cost_func(x, Y, R,feat_num,lambda)),init_val, options);
    toc;

    X = reshape(vec_obj(1:item_num*feat_num), item_num, feat_num);
    %Theta = reshape(vec_obj(item_num*feat_num+1:end),user_num, feat_num);
    Theta = reshape(vec_obj((item_num*feat_num+1):(item_num*feat_num + user_num*feat_num)), user_num, feat_num);       
    b = reshape(vec_obj((item_num*feat_num + user_num*feat_num+ 1) :end),1,user_num);
    P = X * Theta'+ repmat(b, item_num,1);% + repmat(Ymean,1,user_num);
end