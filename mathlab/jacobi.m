% Jacobi method
% INPUT  = A system matrix; b rhs; x0 initial guess; tol tolerance; nmax maximum number of iteration
% OUTPUT = x solution, nit number of iterations, time elapsed time, err final error
function [x, nit, time, err]=jacobi(A,b,x0, tol, nmax)

[M,N]=size(A);
L=length(x0);

if M~=N
    display('Matrix A is not a square matrix');
    return
elseif L~=M
    display('Dimensions of matrix A does not match dimension of initial guess x0');
    return
end
if sum(find(diag(A)==0))>0
    display('At least a diagonal entry is non-zero. The method automatically fails')
    return
end

%extract needed matrices
D = diag(diag(A));
B = D-A;

xold = x0;
xnew = xold + 1;
nit=0;

format long
tic
while norm(xnew-xold, inf)>tol && nit < nmax
    xold = xnew;
    xnew = inv(D) * (B*xold + b);
    nit = nit+1;
end
time = toc;
x=xnew;
err = norm(xnew-xold, inf)/norm(xnew,inf);

return