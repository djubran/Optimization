function [f,g,H] = f2(x,par)
%this function is the function compisition of f2=h(phi)=(sqrt(1+(phi(x))^2))
assert(isvector(x) && length(x) == 3);

input = x(:);
[valuePhi,gradientPhi,hessianPhi] = par.Phi(input);
[valueExp,gradientExp,hessianExp] = par.Exp(valuePhi);
if nargout >= 1
    f = valueExp;
end
if nargout >= 2
    g = gradientExp * gradientPhi;
end
if nargout >= 3
    H =   ...
        hessianExp * (gradientPhi * gradientPhi.') ...
        + ...
        gradientExp * hessianPhi;
end

end

