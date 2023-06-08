% INITIALIZE THE NEURAL NETWORK PROBLEM %
% inputs for the neural net
inputs = [1:10;5:14];

% targets for the neural net
targets = cos(inputs(1,:).^2);

% number of neurons
n = 2;
% create a neural network
net = feedforwardnet(n);

% configure the neural network for this dataset
net = train(net, inputs, targets);

% create handle to the MSE_TEST function, that
% calculates MSE
obj_func = @(x) MSE_func(x, net, inputs, targets)

% Setting the Genetic Algorithms options
ga_options = optimoptions('fmincon','TolFun', 1e-8,'display','iter');

% A ffnn with n neurons requires 3n+1 quantities (weights and biases column
% vector)
% a. n for the input weights
% b. n for the input biases
% c. n for the output weights
% d. 1 for the output bias

% To compute actual number of parameters from the initialised network
X0 = formwb(net,net.b,net.IW,net.LW); % or x0 = getwb(net); or x0 = getx(net);
p = numel(X0);
x = ga(obj_func, p, ga_options)

%%
function mse = MSE_func(x, net, inputs, targets)
% 'x' contains the weights and biases vector
x = formwb(net,net.b,net.IW,net.LW);
net = setwb(net, x');

% Evaluate the net
pred = net(inputs);

% Calculating the mean squared error
mse = sum((pred-targets).^2)/length(pred)
end