function params = BFGS (x,params)
    max_iter = 2000;
    y=f(x);
    func = @avg_err;
    grad = @loss_grad;
    B = eye(length(params));    
    g = grad(x,params);
    d = -B*g;
    alpha = armijo(func,grad,d,x,y,params);
    params_next = params + alpha*d;
    ii=1;
    while (ii <=max_iter && norm(g,2) > eps)
        g_next = grad(x,params_next);
        p = params_next-params;
        q = g_next-g;
        s = B*q;
        tau = s.'*q;
        mu = p.'*q;
        v = p/mu-s/tau;
        if (0.9*g'*d < g_next'*d)
            B = B+ p*p.'/mu -s*s.'/tau +tau*(v*v.');
        end
        d = -B*g_next;
        % armijo 
        alpha = armijo(func,grad,d,x,y,params);        
        params = params_next;
        g = g_next;
        params_next = params + alpha*d;
        ii=ii+1;

    end
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

function g = loss_grad(x,params)
    epsilon = 1e-2;
    base = eye(length(params));
    g = zeros(length(params),1);
    for ii = 1:length(params)
        g(ii) = (avg_err(x,params+ epsilon*base(:,ii))-avg_err(x,params- epsilon*base(:,ii)))/(2*epsilon);
    end
end


function res = F(x,params)
    [b1,b2,b3,W1,W2,W3] = extract_params(params);
    res = W3.' * my_tanh(W2.' * my_tanh(W1.' * x + b1) + b2) +b3;
end
function out = my_tanh(x)
    out = (exp(x)-exp(-x)) ./ (exp(x)+exp(-x)) ;
end
function out = tanh_grad(x)
    out = (4.*exp(-2*x)) ./ (1 + exp(-2*x)).^2;
end
function err = avg_err(x,params)
    y=f(x);
    err = mean((F(x,params)-y).^2);
end
function y = f(x)
    y = x(1,:) .* exp(-x(1,:).^2-x(2,:).^2);
end
function alpha = armijo(func, grad, d, x, y, params)
    sigma = 0.25;
    beta = 0.5;
    alpha = 1;
     
    f0 = func(x,params);
    df0 = (grad(x,params).')*d;

    if(func(x,params+alpha*d)-f0<=sigma*alpha*df0)
        while(func(x,params+alpha*d)-f0<=sigma*alpha*df0)
            alpha = alpha/beta;
        end
        alpha = alpha*beta;
    else
        while(~(func(x,params+alpha*d)-f0<=sigma*alpha*df0))
            if(alpha ==0)
                return
            end
            alpha = alpha*beta;
        end
    end
end
