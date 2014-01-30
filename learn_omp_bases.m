function [ A ] = learn_omp_bases(...
    X, basis_count, omp_num, step, round_count, l1_bases, Ai )
% Learn linear bases for the given patches, using orthogonal matching pursuit.
%
% Parameters:
%   X: observations to use in basis learning (obs_count x obs_dim)
%   basis_count: number of bases to learn
%   omp_num: number of bases with which to represent each observation
%   step: step size for basis gradient descent
%   round_count: number of update rounds to perform
%   l1_bases: l1(ish) regularization weight to apply to bases
%   Ai: optional initial set of bases (obs_dim x basis_count)
% Outputs:
%   A: learned set of bases (obs_dim x basis_count)
%
obs_dim = size(X,2);
if exist('Ai','var')
    if (size(Ai,1) ~= obs_dim || size(Ai,2) ~= basis_count)
        error('learn_omp_bases: mismatched initial basis size.\n');
    end
    A = Ai;
else
    A = randn(obs_dim,basis_count);
    A = bsxfun(@rdivide, A, sqrt(sum(A.^2) + 1e-5));
end

% Do round_count alternations between OMP encoding and basis updating
for r=1:round_count,
    fprintf('ROUND %d\n',r);
    B = omp_encode(X, A, omp_num);
    [ A obj ] = omp_basis_update(X, A, B, l1_bases, step);
end

return
end

function [ B ] = omp_encode( X, A, omp_num )
% Use orthogonal matching pursuit to encode the observations in X using the
% bases in A.
%
% Parameters:
%   X: observations to encode (obs_count x obs_dim)
%   A: bases with which to encode observations (obs_dim x basis_count)
%   omp_num: the number of bases to include in each reconstruction
% Outputs:
%   B: encoding of observations in terms of bases (obs_count x basis_count)
%
obs_count = size(X,1);
basis_count = size(A,2);
A_sqs = sum(A.^2);
Xr = X;
B = zeros(obs_count, basis_count);
B_idx = zeros(obs_count, omp_num);
fprintf('  OMP encoding {\n');
for i=1:omp_num,
    fprintf('    B%d:', i);
    dots = Xr * A;
    scores = dots ./ (sqrt(sum(Xr.^2,2)) * sqrt(A_sqs));
    [max_scores max_idx] = max(abs(scores),[],2);
    for j=1:obs_count,
        if (mod(j,round(obs_count / 50)) == 0)
            fprintf('.');
        end
        idx = max_idx(j);
        B_idx(j,i) = idx;
        w = (Xr(j,:) * A(:,idx)) / A_sqs(idx);
        Xr(j,:) = Xr(j,:) - (A(:,idx)' .* w);
        B(j,idx) = B(j,idx) + w;
    end
    fprintf('\n');
end
obs_var = sum(sum((bsxfun(@minus,X,mean(X,2))).^2));
obj = sum(Xr(:).^2) / obs_var;
fprintf('    obj: %.6f\n', obj);
fprintf('  }\n');

return
end

function [ A_new obj ] = omp_basis_update( X, A, B, lam_l1, step )
% Update the bases in A, given the encoding of the observations in X according
% to the weightts in B. Use gradient descent step size "step".
%
% Parameters:
%   X: observations that were encoded (obs_count x obs_dim)
%   A: bases used in the encoding (obs_dim x basis_count)
%   B: encoding weights (obs_count x basis_count)
%   l1_bases: L1(ish) regularization weight to apply to bases
%   step: gradient descent step size
% Outputs:
%   A: updated bases
%   obj: average squared error of the reconstructions
%

fprintf('  OMP updating {\n');
obs_count = size(X,1);
obs_var = sum(sum((bsxfun(@minus,X,mean(X,2))).^2));

Xh = B * A';
Xr = Xh - X;
obj = sum(Xr(:).^2) / obs_var;
fprintf('    pre_obj: %.6f\n', obj);

A_grad = ((Xr' * B) ./ obs_count) + ((A ./ sqrt(A.^2 + 1e-5)) .* lam_l1);
A_new = A - (A_grad .* step);
A_new = bsxfun(@rdivide,A_new,sqrt(sum(A_new.^2) + 1e-5));
Xh = B * A_new';
obj = sum((X(:) - Xh(:)).^2) / obs_var;
fprintf('    post_obj: %.6f\n', obj);
fprintf('  }\n');

return
end


