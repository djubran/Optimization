function [ xconvergence, fconvergence ] = gradint_descent( func, x , type,Q )
% Gradient Descent with Armijo step size
%type =1 inexact ; 0 exact
% Initializing
n=length(x);
sigma=0.25; 
beta=0.5;
epsilon=10^-5;
a0=1;
k=1;
[f,g]=func(x ,Q);
max_k=10^5;
fconvergence=zeros(max_k,1);
xconvergence=zeros(max_k,n);

% Gradient Descent
while(norm(g)>=epsilon && k<max_k)
    fconvergence(k,1)=f;
    xconvergence(k,:)=x;
    d= -g;    %d= d/ norm(d);
    if type==1
        s=armijo_step(a0,x,func,f,g,d,sigma,beta,Q);
    else
        s=-x/d;
    end
    k=k+1;
    x=x+s*d;
    [f,g]=func(x,Q);
end

fconvergence=fconvergence(1:k-1,:);
xconvergence=xconvergence(1:k-1,:);

end





