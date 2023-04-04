function [value,gradient,hessian] = myH(x)
%this function is the h function in the 4th section of q1 
%R->R; analytic evaluations of the function, it's gradient and hessian
value = (1+x^2)^0.5;
gradient = x /(1+x^2)^0.5;
hessian =1/(1+x^2)^1.5;
end

