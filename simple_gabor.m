clear;
% Draw some simple, quick-to-compute and easily rotated Gabor-like features
[ Xg Yg ] = meshgrid(linspace(-3, 3, 500), linspace(-3, 3, 500));
Pg = [Xg(:) Yg(:)];

% Set up centers for the normal distributions
mu_left = [-1.5 0.0];
mu_center = [0.0 0.0];
mu_right = [1.5 0.0];
sigma = 1.0;
scale_count = 40;
scales = logspace(log10(0.25), log10(4.0), scale_count);

% Compute filter shape at each relative center/surround scale ratio
fprintf('Compting distances...\n');
Fg = zeros(size(Xg,1),size(Xg,2),scale_count);
D_left = sqrt(sum((Pg - repmat(mu_left, size(Pg,1), 1)).^2, 2));
D_center = sqrt(sum((Pg - repmat(mu_center, size(Pg,1), 1)).^2, 2));
D_right = sqrt(sum((Pg - repmat(mu_right, size(Pg,1), 1)).^2, 2));
fprintf('Computing filter shapes:');
for i=1:scale_count,
    sigma_lr = scales(i) * sigma;
    Vg = normpdf(D_center, 0, sigma) - ...
         0.5 * (normpdf(D_left, 0, sigma_lr) + normpdf(D_right, 0, sigma_lr));
    Fg(:,:,i) = reshape(Vg, size(Xg));
    fprintf('.');
end
fprintf('DONE\n');

