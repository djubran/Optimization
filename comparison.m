close all;clear all;clc
%calculate difference between the analytical and numerical
%evaluation of the gradient and Hessian of functions f1 and f2
% produce random vector x ∈ R^3 ;random matrix A ∈ R^3xR^3
x = sort(randn(3,1));x = sort(x);
par.A = magic(3)*0.1;
par.Phi = @myPhi;
par.Exp = @myH;
%% Infinity norm error
% evaluate the infinity norm of the difference between the numerical and analytical gradient and
%Hessian of f1 and f2 at the point x
epsVecN = 2.^(-60:0);
epsVec = flip(epsVecN);%flip vector to match values from 0 to 60(2^0-2^-60) and make it easier to determine which power gets the optimal approximation
gError1 = [];
hError1 = [];
gError2 = [];
hError2 = [];
for idx=1:length(epsVec)
    par.myEps = epsVec(idx);
    [~,par1.g1,par1.H1] = f1(x,par);%analytic evaluation of func1 hessian and gradient
    [~,par2.g2,par2.H2] = f2(x,par);%analytic evaluation of func2 hessian and gradient
    [gnum1,Hnum1] = numdiff(@f1,x,par);%numerical evaluation of func1 hessian and gradient
    [gnum2,Hnum2] = numdiff(@f2,x,par);%numerical evaluation of func2 hessian and gradient
    %% Calculate Gradient infinity norm error
    gError1 = [gError1 max(par1.g1(:)-gnum1(:))];
    gError2 = [gError2 max(par2.g2(:)-gnum2(:))];
    
    figure(1);
    semilogy(gError1);%plot y logrithmically 
    title('F1 Gradient infinity norm error');
    legend('f1');
    xlabel('epsilon');ylabel('error');
    
    figure(2);
    semilogy(gError2);   %plot y logrithmically  
    title('F2 Gradient infinity norm error');
    legend('f2');
    xlabel('epsilon');ylabel('error');
    %% Calculate Hessian infinity norm error
    hError1 = [hError1 max(par1.H1(:)-Hnum1(:))];
    hError2 = [hError2 max(par2.H2(:)-Hnum2(:))];
    figure(3);
    semilogy(hError1);
    title('F1 Hessian infinity norm error');
    legend('f1');
    xlabel('epsilon');ylabel('error');
    figure(4);
    semilogy(hError2);
    title('F2 Hessian infinity norm error');
    legend('f2');
    xlabel('epsilon');ylabel('error');
end
