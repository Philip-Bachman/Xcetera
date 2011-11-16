function [ I_new ] = deshred_old( I_shred )
%% Deshred the vertically shredded image in the matrix I_shred.
I_cols = size(I_shred,2);
I_rows = size(I_shred,1);
col_diffs = zeros(I_cols,1);
Id = double(I_shred);

% Normalize columns to unit color length
for i=1:I_cols,
    mean_norm = mean(sqrt(sum(squeeze(Id(:,i,:)).^2,2)));
    Id(:,i,:) = Id(:,i,:) ./ mean_norm;
end

% Find difference between each column and preceeding column
for i=2:I_cols,
    diff = squeeze(Id(:,i,:) - Id(:,i-1,:));
    col_diffs(i) = sum(sqrt(sum(diff.^2,2))) / I_rows;
end
cd_mean = mean(col_diffs(2:end));
col_diffs(1) = cd_mean;
col_diffs = [repmat([cd_mean],5,1); col_diffs; repmat([cd_mean],5,1)];
col_diffs = col_diffs ./ conv(col_diffs,normpdf(-7:7,0.0,2.0),'same');
col_diffs = col_diffs(6:end-5);

% Compute the mean column difference stride length
strides = 5:105;
stride_count = numel(strides);
stride_steps = zeros(stride_count,1);
stride_diffs = zeros(stride_count,1);
stride_starts = zeros(stride_count,1);
for i=1:stride_count,
    stride = strides(i);
    for j=1:stride,
        stride_idx = j:stride:I_cols;
        stride_mean = mean(col_diffs(stride_idx));
        if (stride_mean > stride_diffs(i))
            stride_diffs(i) = stride_mean;
            stride_steps(i) = numel(stride_idx);
            stride_starts(i) = j;
        end
    end
end

% Bootstrap distributions of mean column difference for each stride length
sample_count = 5000;
stride_bs_diffs = zeros(stride_count,5000);
for i=1:stride_count,
    steps = stride_steps(i);
    rvs = zeros(5000,1);
    for j=1:5000,
        ridx = randsample(I_cols,steps);
        rvs(j) = mean(col_diffs(ridx));
    end
    stride_bs_diffs(i,:) = rvs(:);
end

% For each optimal stride alignment, compute its probability under a Gamma
% distribution, with parameters inferred from the bootstrap distributions
stride_gpdfs = zeros(stride_count,1);
for i=1:stride_count,
    phat = gamfit(stride_bs_diffs(i,:));
    stride_gpdfs(i) = log(gampdf(stride_diffs(i),phat(1),phat(2)));
end

%% Split image into shreds and compute L/R edge match costs between all pairs
[min_score min_idx] = min(stride_gpdfs);
min_start = stride_starts(min_idx);
min_stride = strides(min_idx);

stride_mask = [];
intervals = min_start:min_stride:I_cols;
intervals = [intervals I_cols+1];
for i=2:numel(intervals),
    stride_mask = [stride_mask; (1:I_cols < intervals(i)) & (1:I_cols >= intervals(i-1))];
end

stride_mask = logical(stride_mask);
shred_count = size(stride_mask,1);
match_scores = zeros(shred_count,shred_count);
for i=1:shred_count,
    for j=1:shred_count,
        if (i ~= j)
            shred_1 = Id(:,stride_mask(i,:),:);
            shred_2 = Id(:,stride_mask(j,:),:);
            match_scores(i,j) = match_score(shred_1,shred_2);
            match_scores(j,i) = match_score(shred_2,shred_1);
        end
    end
end

fprintf('AAAAA\n');

I_new = I_shred(:,:,:);

return

end

function [ I_new ] = unmix_shreds(Id, stride_mask)
%%
stride_mask = logical(stride_mask);
shred_count = size(stride_mask,2);
match_scores = zeros(shred_count,shred_count);
for i=1:shred_count,
    for j=1:shred_count,
        if (i ~= j)
            shred_1 = Id(:,stride_mask(i,:),:);
            shred_2 = Id(:,stride_mask(j,:),:);
            match_scores(i,j) = match_score(shred_1,shred_2);
            match_scores(j,i) = match_score(shred_2,shred_1);
        end
    end
end

return

end

function [ score ] = match_score(shred_left, shred_right)
%%
shred_join = cat(2,shred_left,shred_right);
shred_diffs = zeros(size(shred_join,2),1);
for i=2:size(shred_join,2),
    diff = squeeze(shred_join(:,i,:) - shred_join(:,i-1,:));
    shred_diffs(i) = sum(sqrt(sum(diff.^2,2))) / size(shred_join,1);
end
shred_diffs(1) = mean(shred_diffs);
shred_diffs = shred_diffs ./ conv(shred_diffs,normpdf(-7:7,0.0,2.0),'same');
score = shred_diffs(size(shred_left,2) + 1);

return

end

