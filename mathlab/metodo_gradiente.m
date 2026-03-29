% implementazione del metodo del gradiente
% INPUT  : A=system matrix, b=rhs, x0=initial guess,
%          tol=tolerance, nmax= maximum number of iteration
% OUTPUT : xk=solution, nit=number of iteration, time=time elapsed, err=final tolerance
function [xk, nit, time, err]=metodo_gradiente(A,b,x0,tol,nmax)

%check for the properties of matrix A
[M,N]=size(A);
L=length(x0);
if M~=N
    display('Matrix A is not a square matrix');
    return
elseif L~=M
    display('Dimensions of matrix A does not match dimension of initial guess x0');
    return
elseif sum(find(eig(A)<0)) %if the matrix is not positive definite
    display('Matrix A is not positive definite');
    return
elseif issymmetric(A)==0  %if the matrix is not positive definite
    display('Matrix A is not symmetric');
    return
end

nit=0; err=1; xold=x0;
tic
while nit<nmax && err>tol
    residual = b- A*xold; %update of the residual
    step     = (residual'*residual)/(residual'*A*residual); % computation of the step-lenght
    xnew     = xold + step*residual;
    err      = norm(b - A*xnew)/norm(xnew);
    xold     = xnew;
    nit      = nit+1;
end
time=toc;
xk=xnew;

end