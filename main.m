clear ; close all; clc


%% =========== Part 1: Loading and Visualizing Data =============
input_layer_size  = 400;
hidden_layer_size = 400;
X=load('trainingSetsX.txt');
Y=load('trainingSetsY.txt');
X=X'(:);
Y=Y'(:);
%imagesc(Y),colorbar,colormap gray;
%imagesc(X),colorbar,colormap gray;
m = size(X, 1)/400;%the number of traning examples
X=(reshape(X,400,m))';
Y=(reshape(Y,400,m))';

%% ================ Part 2: Loading Parameters ================
theta1=randn(400,401)./10;
theta2=randn(400,401)./10;
theta3=randn(400,401)./10;
theta4=randn(400,401)./10;
nn_params= [theta1(:) ; theta2(:);theta3(:);theta4(:)];% Unroll parameters 

%% ================ Part 3: Compute Cost (Feedforward) ================
lambda = 1;
num_labels=400;


%[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                   num_labels, X, Y, lambda);
                   
%fprintf(['J= %f \n'], J);
%nn_params=nn_params-grad.*0.01;
%[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                   num_labels, X, Y, lambda);
%fprintf(['J= %f \n'], J); 



for i=1:200
  
  [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, Y, lambda);
  nn_params=nn_params-grad.*0.01;
  fprintf(['J= %f \n'], J); 
end


%initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
%initial_Theta3 = randInitializeWeights(input_layer_size, hidden_layer_size);
%initial_Theta4 = randInitializeWeights(hidden_layer_size, num_labels);
%initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:);initial_Theta3(:) ; initial_Theta4(:)];
%  
%options = optimset('MaxIter', 50);
%
%%  You should also try different values of lambda
%lambda = 1;
%
%% Create "short hand" for the cost function to be minimized
%
%costFunction = @(p) nnCostFunction(p, ...
%                                   input_layer_size, ...
%                                   hidden_layer_size, ...
%                                   num_labels, X, Y, lambda);
%
%% Now, costFunction is a function that takes in only one argument (the
%% neural network parameters)
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%  
%  
%  
  
TESTX=load('test.txt');
TESTX=TESTX'(:);
TESTX=(reshape(TESTX,400,1))';
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):(2*(hidden_layer_size * (input_layer_size + 1)))), ...
                 hidden_layer_size, (hidden_layer_size + 1));
Theta3 = reshape(nn_params((1 + 2*(hidden_layer_size * (input_layer_size + 1))):(3*(hidden_layer_size * (input_layer_size + 1)))), ...
                 num_labels, (hidden_layer_size + 1));
Theta4 = reshape(nn_params((1 + 3*(hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));            
                 
a1 = [ones(1, 1) TESTX]; 
 
z2 = a1 * Theta1';  
a2 = sigmoid(z2);  
a2 = [ones(1, 1) a2];

z3 = a2 * Theta2'; 
a3= sigmoid(z3);
a3=[ones(1,1) a3];

z4 = a3 * Theta3'; 
a4= sigmoid(z4);
a4=[ones(1,1) a4];

z5=a4 * Theta4'; 
h = sigmoid(z5);
h=reshape(h,20,20);
h=h';
imagesc(h),colorbar,colormap gray;