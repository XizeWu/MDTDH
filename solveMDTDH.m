function [P1, P2, theta1, theta2] = solveMDTDH(X1, X2, para)
% X1        the nonlinear representation of the image modality
% X2        the nonlinear representation of the text modality
% omega1    the weight of the image modality
% omega2    the weight of the text modality
% theta1    the weight of the image modality
% theta2    the weight of the text modality
% U1        the basis matrix of the image modality
% U2        the basis matrix of the text modality
% P1        the projection matrix of the image modality
% P2        the projection matrix of the test modality
% B         hash codes

    %% Parameters setting
    beta = para.beta;
    lambda = para.lambda;
    bits = para.bits;
    MAX_iter = para.MAX_iter;
    iter=1;
    t=2;

    %% Random initialization
    [row1, col1] = size(X1);
    [row2, ~] = size(X2);
    U1 = randn(row1, bits);
    [PP1, ~, QQ1] = svd(U1, 'econ');
    U1 = PP1 * QQ1';

    U2 = randn(row2, bits);
    [PP2, ~ , QQ2] = svd(U2, 'econ');
    U2 = PP2 * QQ2';

    P1 = randn(bits, row1);
    P2 = randn(bits, row2);
    B = sign(randn(bits, col1));

    %% Iteration
    while (true)	
        %% Update omega1,omega2
        h1 = sum(sum((X1 - U1 * B).^2));
        h2 = sum(sum((X2 - U2 * B).^2));
        hh1 = (1/h1).^(1/(t-1));
        hh2 = (1/h2).^(1/(t-1));
        omega1 = hh1 / (hh1 + hh2);
        omega2 = hh2 / (hh1 + hh2); 

        %% Update U1 and U2
        G1 = 2 * omega1 * X1 * B';
        [PP1, ~, QQ1] = svd(G1, 'econ');
        U1 = PP1 * QQ1';
        G2 = 2 * omega2 * X2 * B';
        [PP2, ~, QQ2] = svd(G2, 'econ');
        U2 = PP2 * QQ2';

        %% Update theta1 and theta2
        j1 = sum(sum((B - P1 * X1).^2)) ;
        j2 = sum(sum((B - P2 * X2).^2)) ;
        jj1 = (1/j1).^(1/(t-1));
        jj2 = (1/j2).^(1/(t-1));
        theta1 = jj1 / (jj1+jj2);
        theta2 = jj2 / (jj1+jj2); 

        %% Update P1 and P2 
        P1 = B * X1' / (X1 * X1' + (lambda/(beta * theta1)) * eye(row1));
        P2 = B * X2' / (X2 * X2' + (lambda/(beta * theta2)) * eye(row2));

        %% Update B
        B = sign(omega1 * U1'* X1 + omega2 * U2' * X2 + beta * (theta1 * P1 * X1 + theta2 * P2 * X2));

        %% Loss
        currentLoss = omega1 * (norm(X1 - U1 * B, 'fro')^2) + omega2 * (norm(X2 - U2 * B, 'fro')^2) ...
            + beta * theta1 * (norm(B - P1 * X1, 'fro')^2) + beta * theta2 * (norm(B - P2 * X2, 'fro')^2) ...
            + lambda * (norm(P1, 'fro')^2 + norm(P2, 'fro')^2);
        fprintf('iter: %d, currentLoss= %.1f\n', iter, currentLoss);
        if iter >= MAX_iter
            break;
        end
        iter = iter + 1;
    end
end
