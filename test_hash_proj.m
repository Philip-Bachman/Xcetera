% Test the relative effect of (kurtosis of) feature weight distribution on the
% quality of learned function when features are "projected/hashed" to a lower
% dimensional space.
clear;
test_count = 20;
obs_dim = 1000;
obs_count = 25000;
train_size = 20000;
test_size = obs_count - train_size;
hash_factor = 5;
hash_dim = round(obs_dim / hash_factor);
noise_std = 1e-2;
lambda = eye(obs_dim) .* 1e-4;
feat_freqs = 0.1:-0.01:0.01;
freq_count = numel(feat_freqs);

% In each test generate three sets of "true" feature weights.
%   1. Weights distributed like doubly-rectified exponential (aka Laplace).
%   2. Weights distributed like Normal (aka Gaussian)
%   3. Weights distributed like uniform
% For each distribution, scale weights to have unit standard deviation.
test_err = zeros(freq_count, test_count, 4);
for t_num=1:test_count,
    fprintf('Test %d commencing...\n',t_num);
    % Generate a random hash/projection to 1/5 the natural dimension
    H = zeros(obs_dim, hash_dim);
    for i=1:obs_dim,
        H(i,randi(hash_dim)) = 1;
    end
    for f_num=1:freq_count,
        feat_freq = feat_freqs(f_num);
        w_scales = (20 * sign(randn(obs_dim,1))) .* (rand(obs_dim,1) < feat_freq);
        w_scales = w_scales + randn(obs_dim,1);
        % Draw random feature weights from each distribution
        w_d = randn(obs_dim, 1);
        w_s = w_scales;
        % Normalize feature weights to unit variance
        w_d = w_d ./ std(w_d);
        w_s = w_s ./ std(w_s);
        % Sample a set of observations and their hashed/projected representation
        X = randn(obs_count,obs_dim) .* (rand(obs_count,obs_dim) < feat_freq);
        X = bsxfun(@times, X, w_scales');
        Xh = X*H;
        Y_d = X*w_d + (randn(obs_count,1) .* noise_std);
        Y_s = X*w_s + (randn(obs_count,1) .* noise_std);
        % Split observations into training and testing sets
        train_idx = randsample(obs_count,train_size);
        test_idx = setdiff(1:obs_count, train_idx);
        X_train = X(train_idx,:);
        Xh_train = Xh(train_idx,:);
        X_test = X(test_idx,:);
        Xh_test = Xh(test_idx,:);
        % Do basic L2-regularized linear regression on training set
        XtXi = (X_train' * X_train) + lambda;
        b_d = XtXi \ (X_train' * Y_d(train_idx));
        b_s = XtXi \ (X_train' * Y_s(train_idx));
        XhtXhi = (Xh_train' * Xh_train) + lambda(1:hash_dim,1:hash_dim);
        bh_d = XhtXhi \ (Xh_train' * Y_d(train_idx));
        bh_s = XhtXhi \ (Xh_train' * Y_s(train_idx));
        % Measure prediction error for the estimated weights
        E_d = sum((X_test*b_d  - Y_d(test_idx)).^2) / sum(Y_d(test_idx).^2);
        E_s = sum((X_test*b_s  - Y_s(test_idx)).^2) / sum(Y_s(test_idx).^2);
        Eh_d = sum((Xh_test*bh_d  - Y_d(test_idx)).^2) / sum(Y_d(test_idx).^2);
        Eh_s = sum((Xh_test*bh_s  - Y_s(test_idx)).^2) / sum(Y_s(test_idx).^2);
        % Record test errors
        test_err(f_num,t_num,1) = E_d;
        test_err(f_num,t_num,2) = E_s;
        test_err(f_num,t_num,3) = Eh_d;
        test_err(f_num,t_num,4) = Eh_s;
    end
end
