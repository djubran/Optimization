function [f,g,H] = f1(x,par)
%this function is the function compisition of f1=phi(Ax)
assert(isvector(x) && length(x) == 3);
A = par.A;
input = A * x(:);
[value,gradient,hessian] = par.Phi(input);
if nargout >= 1
    f = value;
end
if nargout >=2
    g = A.' * gradient;
end
if nargout >= 3
    H = A.' * hessian * A;
end

end

