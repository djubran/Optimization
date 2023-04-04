function [gnum,Hnum] = numdiff(myfunc,x,par)
%input a function reference f (x, . . .), a vector x ∈ R^N
%a scalar ε ∈ R inside the par-struct
%  returns numerical evaluation of the gradient and Hessian at the point x of the input function f
assert(isvector(x));

x = x(:);

myEps = par.myEps;
e = eye(length(x));

if nargout >=1
    gnum = zeros(length(x),1);
    for n=1:length(x)
        fplus = myfunc(x + myEps*e(:,n),par);
        fminus = myfunc(x - myEps*e(:,n),par);
        gnum(n) = (1/2/myEps) * (fplus  - fminus);
    end
end
if nargout >= 2
    Hnum = zeros(length(x));
    for n=1:length(x)
        [~,gplus] = myfunc(x + myEps*e(:,n),par);
        [~,gminus] = myfunc(x - myEps*e(:,n),par);
        Hnum(:,n) = (1/2/myEps) * (gplus  - gminus); 
    end
end

end
