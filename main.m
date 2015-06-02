clc;
clear;
lambdas = [0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24,20.48,40.96,81.92];
feat = [5,10,15,20,30,40];
load('test_all.mat');
load('train_all.mat');
R = L_train;
Y = Rating_train;
error_rate = zeros(length(lambdas),length(feat));
for i=1:length(lambdas)
    lambda = lambdas(i);
    for j=1:length(feat)
        % cross validation    

        [item_num,user_num] = size(R);
        %lambda = 0.02;
        %feat_num = 10;
        feat_num = feat(j);
        P = promf_resys_func( Y,R,feat_num,lambda);

        error_rate(i,j) = sum(sum((test_R.*P - test_Y).^2))/sum(sum(test_Y.^2));

        fprintf('lambda %f | featnum %d |error: %f\n', lambda, feat_num, error_rate(i,j));
    end
end
%plot(lambdas,error_rate,'-o');
figure;
surf(lambdas, feat, error_rate');
zlabel('error_rate');
ylabel('feat');
xlabel('\lambda');
set(gca,'xscale','log');
set(gcf, 'Color', 'w');

%export_fig lambda_select.eps
