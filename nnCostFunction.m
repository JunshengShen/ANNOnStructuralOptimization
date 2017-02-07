function [J grad]=nnCostFunction(nn_params, input_layer_size, ...
  hidden_layer_size,num_labels, X, Y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (hidden_layer_size + 1))):(2*(hidden_layer_size * (hidden_layer_size + 1)))), ...
                 hidden_layer_size, (hidden_layer_size + 1));
Theta3 = reshape(nn_params((1 + 2*(hidden_layer_size * (hidden_layer_size + 1))):(3*(hidden_layer_size * (hidden_layer_size + 1)))), ...
                 hidden_layer_size, (hidden_layer_size + 1));                
Theta4 = reshape(nn_params((1 + 2*(hidden_layer_size * (hidden_layer_size + 1))+hidden_layer_size * (input_layer_size + 1)):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));
X1=[ones(m,1) X];


a1 = [ones(m, 1) X];  

z2 = a1 * Theta1';    
a2 = sigmoid(z2);  
tmp2 = [ones(m, 1) a2];

z3 = tmp2 * Theta2';
a3 = sigmoid(z3);
tmp3 = [ones(m, 1) a3];

z4 = tmp3 * Theta3'; 
a4 = sigmoid(z4);
tmp4 = [ones(m, 1) a4];

z5 = tmp4 * Theta4'; 
h = sigmoid(z5);
%Implement forward propagation

J = (1/m)* sum(sum(((-Y) .* log(h) - (1 - Y) .* log(1 - h))));
Theta1_new=Theta1(:,2:size(Theta1,2));  
Theta2_new=Theta2(:,2:size(Theta2,2));  
Theta3_new=Theta3(:,2:size(Theta3,2));
Theta4_new=Theta4(:,2:size(Theta4,2));
J=J+lambda/2/m*(Theta1_new(:)'*Theta1_new(:)+Theta2_new(:)'*Theta2_new(:)+Theta3_new(:)'*Theta3_new(:)+Theta4_new(:)'*Theta4_new(:));  
%Implement the cost function



for i=1:m  
    y_new=Y(i,:);  
    a1=[1;X(i,:)'];  
    a2=[1;sigmoid(Theta1*a1)];
    a3=[1;sigmoid(Theta2*a2)];
    a4=[1;sigmoid(Theta3*a3)];
    a5=sigmoid(Theta4*a4);
    det5=a5-y_new';  
    %temp=z4(i,:)';
    det4= Theta4'*det5.*sigmoidGradient(a4); 
    det4=det4(2:end);
    %temp=z3(i,:)';
    det3= Theta3'*det4.*sigmoidGradient(a3);
    det3=det3(2:end);
    %temp=z2(i,:)';
    det2= Theta2'*det3.*sigmoidGradient(a2);  %todo is et3(2:end) right?
    det2=det2(2:end);
    
    Theta1_grad=Theta1_grad+det2*a1';  
    Theta2_grad=Theta2_grad+det3*a2';  
    Theta3_grad=Theta3_grad+det4*a3';
    Theta4_grad=Theta4_grad+det5*a4';
end  
%step 3 and 4  
Theta1_grad(:,1)=Theta1_grad(:,1)/m;  
Theta1_grad(:,2:size(Theta1_grad,2))=Theta1_grad(:,2:size(Theta1_grad,2))/m+...  
    lambda*Theta1(:,2:size(Theta1,2))/m;  
    
Theta2_grad(:,1)=Theta2_grad(:,1)/m;  
Theta2_grad(:,2:size(Theta2_grad,2))=Theta2_grad(:,2:size(Theta2_grad,2))/m+...  
    lambda*Theta2(:,2:size(Theta2,2))/m;  

Theta3_grad(:,1)=Theta3_grad(:,1)/m;  
Theta3_grad(:,2:size(Theta3_grad,2))=Theta3_grad(:,2:size(Theta3_grad,2))/m+...  
    lambda*Theta3(:,2:size(Theta3,2))/m;     
    
Theta4_grad(:,1)=Theta4_grad(:,1)/m;  
Theta4_grad(:,2:size(Theta4_grad,2))=Theta4_grad(:,2:size(Theta4_grad,2))/m+...  
lambda*Theta4(:,2:size(Theta4,2))/m;     
%Implement backpropagation



grad = [Theta1_grad(:) ; Theta2_grad(:);Theta3_grad(:);Theta4_grad(:)];


end


