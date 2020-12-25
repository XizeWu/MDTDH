function [B] = queryMDTDH(X1, X2, P1,P2,theta1,theta2,bits)
MaxIter=10;
iter = 1;
LastLoss = 99999999;
num = size(X1,2);
t=2;
B = sign(randn(bits,num));

while (iter<=MaxIter)
	j1 = sum(sum((B - P1 * X1).^2));
    j2 = sum(sum((B - P2 * X2).^2));
    jj1 = (1/j1).^(1/(t-1));
    jj2 = (1/j2).^(1/(t-1));
	theta1 = jj1 / (jj1+jj2);
    theta2 = jj2 / (jj1+jj2); 
    B = sign(theta1 * P1 * X1 + theta2 * P2 * X2);
    
    Loss = theta1*norm(B - P1 * X1, 'fro')^2 + theta2*norm(B - P2 * X2, 'fro')^2 ;
    deltaLoss = LastLoss - Loss;
    LastLoss = Loss;
    fprintf('QueryIter:%d | QueryLoss= %.4f\n',iter,Loss);
    if iter>3 && deltaLoss < 0.001
        break;
    end
    iter = iter+1;
end
end
