
%% BFGS - Rosenbrock function
n = 10;
x0 = zeros(n,1);
f_star = 0;
f  = BFGS_Rosenbrock(@rosenbrock, @rosenbrock_grad, x0);
figure(1);
title('BFGS - Rosenbrock function');
semilogy(f - f_star);
xlabel('k');
ylabel('f(x_{k}) - f^{*}');


function m = BFGS_Rosenbrock( fun, fun_gradient, x0 )
    epsilon = 10^(-5);
    alpha0 = 1;
    sigma = 0.25;
    beta = 0.5;
    max_itr = 2000; 
    x = x0;
    f = zeros(max_itr,1);
    index = 1;
    f(index) = fun(x);
    g = fun_gradient(x);
    B = eye(size(x,1));
    % Keep going till ||g(x)|| < epsilon
    while (norm(g,2) > epsilon && index <= max_itr)
        d = -B*g;
        alpha = armijo(alpha0, x, d, fun, fun_gradient, sigma, beta);
        x_next = x + alpha*d;
        
        index = index + 1;
        f(index) = fun(x_next);
        g_next = fun_gradient(x_next);
        
        if (g_next.'*d > 0.9.*g.'*d)
           p = x_next - x;
           q = g_next - g;
           s = B*q;
           tau = s.'*q;
           mu = p.'*q;
           v = p / mu - s / tau;
           B = B + p*p.'/mu - s*s.'/tau + tau*v*v.';
        end
        
        g = g_next;
        x = x_next;
    end
    
    % Remove unnecessary zeros from the end of the error_vec
    m = f(1 : index - 1);

end
function [alpha_armijo] = armijo(alpha,x,d,f0,f1,sigma,beta)
%% ARMIJO
% DESCRIPTION:
% function to check whether the provided steplength satisfies Armijo 
% condition: f(x+alpha*d)<=f(x)+gamma*alpha*gradient(f(x))'*d
% if this condition is not met, it extracts the greatest value of alpha that
% satisfies Armijo's expression.
%
% function [alpha_armijo] = armijo(alpha,x,d,f0,f1,gamma,delta)
% INPUT:
%       NOTE: (*) indicates necessary input, the other variables are optional 
%       (*) alpha     - current steplength (1*1);
%       (*) x         - current iterate    (N*1);
%       (*) d         - search direction   (N*1);
%           gamma     - constant provided by the user (1*1) into the range [0,0.5]
%           delta     - constant provided by the user (1*1) into the range [0,  1]
%       (*) f0        - function handle of the objective function          (RN->R );
%       (*) f1        - the gradient (as function handle) of the function  (RN->RN);
% OUTPUT:
%       alpha_armijo - value of alpha whether the condition holds      (1*1);
% REVISION:
%       Ennio Condoleo Rome, Italy h: 23.04 9 Jan 2014
    if (nargin<6)
        beta = 0.5;
        sigma = 1e-4;
    elseif (nargin==6)
        beta = 0.5;
    end
    
    j    = 1;
    while (j>0)
        x_new = x+alpha.*d;
        if (f0(x_new)<=f0(x)+sigma*alpha*f1(x)'*d)
            j = 0;
            alpha_armijo = alpha;
        else
            alpha = alpha*beta;
        end    
    end

end
function [f] = rosenbrock(x);

D = length(x);
f = sum( (1-x(1:D-1)).^2+100*(x(2:D)-x(1:D-1).^2).^2);
end
function [df] = rosenbrock_grad(x);

D = length(x);

df = zeros(D, 1);
df(1:D-1) = - 400*x(1:D-1).*(x(2:D)-x(1:D-1).^2) - 2*(1-x(1:D-1));
df(2:D) = df(2:D) + 200*(x(2:D)-x(1:D-1).^2);
end