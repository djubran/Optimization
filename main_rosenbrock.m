n=10;
x_start=zeros(n,1);
% %% rosenbrock Gradient Descent
[~,f]=gradient_descent(@rosenbrock,x_start, 1,1); %type =1inexact ; 0 exact
gd_rosenbrock=f-0;%  f* of rosenbrock function =0
% Gradient Descent
figure;
semilogy(1:length(gd_rosenbrock),gd_rosenbrock);
xlabel('Iteration ');
ylabel('f(x_k)-f*');
title('Gradient Descent Rosenbrock function');
%% rosenbrock Newton Method
[~,f]=newton_method(@rosenbrock, x_start,1,1);
nm_rosenbrock=f-0;%  f* of rosenbrock function =0
% Newton Method plot
figure;
semilogy(1:length(nm_rosenbrock),nm_rosenbrock);
xlabel('Iteration');
ylabel('f(x_k)-f*');
title('Newton Method Rosenbrock function');





% %dont forget to deleteeee
% Qudratic FUNCTION 
% % %% plot  function convergence in 2D
% %set1
% n=2;
% x_start=[-0.2; -2];
% Q = [10 0 ;0 1];
% [x_nm,f]=newton_method(@quadratic,x_start,1,Q);
% figure;
% x = linspace(-2,2);
% y = linspace(-2,2);
% [xx,yy] = meshgrid(x,y); 
% ff = f(xx,yy);
% figure;
% subplot(2,1,1);
% contour(x,y,f);
% colorbar; hold on;
% plot(x_nm(:,1),x_nm(:,2),'r','LineWidth',2);
% title('set 1 newton function 2D')
% 
% subplot(2,1,2);
% [x_gd,~]=gradint_descent(@rosenbrock,[-2;2]);
% [x_nm,~]=newton_method(@rosenbrock,[-2;2]);
% surfc(ff);hold on;
% plot(x_gd(:,1),x_gd(:,2),'k','LineWidth',2);
% plot(x_nm(:,1),x_nm(:,2),'r','LineWidth',2);
