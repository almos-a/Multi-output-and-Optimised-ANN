clc
clear
close all
%%
% Load some data
load('ExpData.mat');
[m,n] = size(Data) ;

% Split into train and test
P = 0.85;
Training = Data(1:round(P*m),:) ; 
Testing = Data(round(P*m)+1:end,:);
XTrain = Training(:,1:n-1);
YTrain = Training(:,n);
XTest = Testing(:,1:n-1);
YTest = Testing(:,n);
%% Optimization approach
% Define a train/validation split to use inside the objective function
cv = cvpartition(numel(YTrain), 'HoldOut', 0.18); % or using Kfold splits data to an equal partition

% Define hyperparameters to optimize
vars = [optimizableVariable('hiddenLayerSize', [1,27], 'Type', 'integer','Optimize',true);
	    optimizableVariable('lr', [1e-3 1], 'Optimize', true,'Transform','log')];

% Optimize
minfn = @(T)kfoldLoss(XTrain', YTrain', cv, T.hiddenLayerSize, T.lr);
results = bayesopt(minfn, vars,'IsObjectiveDeterministic', false,...
    'AcquisitionFunctionName', 'expected-improvement-plus');
T = bestPoint(results)

% Train final model on full training set using the best hyperparameters
net = feedforwardnet(T.hiddenLayerSize, 'traingd'); % using gradient descent. It can be changed
net.trainParam.lr = T.lr;
net = train(net, XTrain', YTrain');

% Evaluate on test set and compute final rmse
ypred = net(XTest');
finalrmse = sqrt(mean((ypred - YTest').^2))

%% Helper functions

function rmse = kfoldLoss(x, y, cv, numHid, lr)
% Train net.
net = feedforwardnet(numHid, 'traingd');
net.trainParam.lr = lr;
net = train(net, x(:,cv.training), y(:,cv.training));
% Evaluate on validation set and compute rmse
ypred = net(x(:, cv.test));
rmse = sqrt(mean((ypred - y(cv.test)).^2));
end