function [J,grad] = cost_func(obj,Y,R,feat_num,lambda )
%COST_FUNC Summary of this function goes here
%   Detailed explanation goes here
%   J:      cost
%   grad:   gradient
    
    [item_num,user_num] = size(R);
    X = reshape(obj(1:item_num*feat_num), item_num, feat_num);
    %Theta = reshape(obj(item_num*feat_num+1:end), user_num, feat_num);
    Theta = reshape(obj((item_num*feat_num+1):(item_num*feat_num + user_num*feat_num)), user_num, feat_num);       
    b = reshape(obj((item_num*feat_num + user_num*feat_num+ 1) :end),1,user_num);
    Offset = repmat(b, item_num,1);
    J = 1/2*sum(sum((R.*(X*Theta' + Offset -Y)).^2)) + lambda/2*sum(sum(Theta.^2)) + lambda/2*sum(sum(X.^2))+ lambda/2*sum(b.^2);

    X_grad = R.*(X*Theta'+ Offset - Y)*Theta + lambda*X;
    Theta_grad = (R.*(X*Theta'+ Offset - Y))'*X + lambda*Theta;
    Offset_grad = sum(R.*(X*Theta'+ Offset - Y)) + lambda*b;
    grad = [X_grad(:); Theta_grad(:); Offset_grad(:)];
end

