[b1,b2,b3,W1,W2,W3] = intilaize();
par = struct('tanh',@my_tanh,'tanh_grad',@tang_grad,'f',@f,'F',@F,'W1',W1,'W2',W2,'W3',W3,'b1',b1,'b2',b2,'b3',b3);
% creating train and test sets
Ntrain=500;
[x_train,y_train]=create_train(Ntrain);
Ntest=200;
[x_test,y_test] = create_test(Ntest);
params = [reshape(b1,[],1); reshape(b2,[],1); reshape(b3,[],1); reshape(W1,[],1);...
    reshape(W2,[],1); reshape(W3,[],1)];
params = BFGS(x_train,params);
visual(x_test,y_test,params,true);
function [b1,b2,b3,W1,W2,W3] = visual(x,y,params,b)
[b1,b2,b3,W1,W2,W3] = extract_params(params);
par = struct('tanh',@my_tanh,'tanh_grad',@tang_grad,'f',@f,'F',@F,'W1',W1,'W2',W2,'W3',W3,'b1',b1,'b2',b2,'b3',b3);
network_reconstruction = F(x,par).';
[X1, X2] = meshgrid(-2:0.2:2, -2:0.2:2);
Y = X1 .* exp(-X1.^2 - X2.^2);
figure; surf(X1, X2, Y);hold on;
title('Ground Truth Function and the scatter of dataset; epsilon=10e-4')
if b==true 
    scatter3(x(1,:),x(2,:),network_reconstruction,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5);
end
end
function [b1,b2,b3,W1,W2,W3] = intilaize()
b1 = zeros(4,1); 
b2 = zeros(3,1); 
b3=0;
W1 = randn(2,4)/sqrt(2);
W2 = randn(4,3)/sqrt(4);
W3 = randn(3,1)/sqrt(3);
end
function [x_train,y_train] = create_train(n)
Ntrain=n;
x_train= 4*rand(2,Ntrain)-2;
y_train = f(x_train);
end
function [x_test,y_test] = create_test(n)
Ntest=n;
x_test= 4*rand(2,Ntest)-2;
y_test = f(x_test);
end

%% Functions
% ground truth y function
function y = f(x)
    y = x(1,:) .* exp(-x(1,:).^2-x(2,:).^2);
end

% tanh
function out = my_tanh(x)
    out = (exp(x)-exp(-x)) ./ (exp(x)+exp(-x)) ;
end

function out = tanh_grad(x)
    out = (4.*exp(-2*x)) ./ (1 + exp(-2*x)).^2;
end
function res = F(x,par)
    res = par.W3.' * par.tanh(par.W2.' * par.tanh(par.W1.' * x + par.b1) + par.b2) +par.b3;
end
function res = loss_t(x,par)
    res = 2*(par.F(x,par)-par.f(x));
end

function [W1_g,b1_g,W2_g,b2_g,W3_g,b3_g] = grads(x,par)
    loss_derv =loss_t(x,par);
    q1 = par.my_tanh(par.W1.' * x + par.b1);
    q2 = par.my_tanh(par.W2.' * q1 + par.b2);
    
    b3_g = loss_derv;
    W3_g = loss_derv*q2;    
    b2_g = loss_derv * par.W3.' * diag(par.tanh_grad(par.W2.'*q1+par.b2));
    W2_g = loss_derv*q1 * par.W3.' * diag(par.tanh_grad(par.W2.'*q1+par.b2));
    b1_g = loss_derv * par.W3.' * diag(par.tanh_grad(par.W2.' * par.my_tanh(par.W1.'*x + par.b1) + par.b2)) * par.W2.' * diag(par.tanh_grad(par.W1.'*x+par.b1));
    W1_g = loss_derv * x * par.W3.' * diag(par.tanh_grad(par.W2.' * par.tanh_grad(par.W1.'*x + par.b1) + par.b2)) * par.W2.' * diag(par.tanh_grad(par.W1.'*x+par.b1));  
end
function err = avg(x,params)
    err = mean((grads(x,params)-y).^2);
end

function [b1,b2,b3,W1,W2,W3] = extract_params(params)
    start = 1;
    finish = 4;
    b1 = params(start:finish);
    start = finish+1;
    finish = finish+3;
    b2 = params(start:finish);
    start = finish+1;
    finish = finish+1;
    b3 = params(start:finish);
    start = finish+1;
    finish = finish+8;
    W1 = reshape(params(start:finish),2,4);
    start = finish+1;
    finish = finish+12;
    W2 = reshape(params(start:finish),4,3);
    start = finish+1;
    finish = finish+3;
    W3 = reshape(params(start:finish),3,1);
    
end
