function [ I ] = deshred_new( I_shred, opt_shred_count )
% Deshred the vertically shredded image in the matrix I_shred.
I_cols = size(I_shred,2);
I_rows = size(I_shred,1);
col_diffs = zeros(I_cols,1);
Id = double(I_shred);

if ~exist('opt_shred_count','var')
    opt_shred_count = 0;
end

% Find difference between each column and preceeding column
for i=2:I_cols,
    diff = squeeze(Id(:,i,:) - Id(:,i-1,:));
    col_diffs(i) = sum(sqrt(sum(diff.^2,2))) / I_rows;
end
col_diffs(1) = col_diffs(2);
% Compute a distribution for what are probably intra-shred normalized diffs
cd_small = col_diffs(col_diffs < quantile(col_diffs,0.9));
cd_small = cd_small ./ conv(cd_small,normpdf(-7:7,0,2.0),'same');
cd_small = cd_small(6:end-6);
cds_mu = mean(cd_small);
cds_sigma = std(cd_small);
% Compute the normalized inter-column diffs
raw_diffs = col_diffs(:);
col_diffs = [repmat(col_diffs(1),5,1); col_diffs; repmat(col_diffs(end),5,1)];
col_diffs = col_diffs ./ conv(col_diffs,normpdf(-7:7,0.0,2.0),'same');
col_diffs = col_diffs(6:end-5);
[cd_sort cd_sort_idx] = sort(col_diffs,'descend');
% Remove from the sorted list any values too close to earlier values
cds_old = cd_sort(:);
cds_idx_old = cd_sort_idx(:);
cd_sort = [];
cd_sort_idx = [];
for i=1:numel(cds_idx_old),
    if (min(abs([cd_sort_idx 1 I_cols+1] - cds_idx_old(i))) >= 5)
        cd_sort = [cd_sort cds_old(i)];
        cd_sort_idx = [cd_sort_idx cds_idx_old(i)];
    end
end

% Compute the mean column difference for each shred count
shred_counts = 5:50;
sc_count = numel(shred_counts);
sc_diffs = zeros(sc_count,1);
sc_gpdfs = zeros(sc_count,1);
for i=1:sc_count,
    sc_diffs(i) = mean(cd_sort(1:shred_counts(i)));
end

% Compute an optimal shred count, if one was not given as a parameter
if (opt_shred_count == 0)
    % Bootstrap distributions of mean column difference for each shred count
    sample_count = 10000;
    sc_bs_diffs = zeros(sc_count,sample_count);
    for i=1:sc_count,
        shreds = shred_counts(i);
        rvs = zeros(sample_count,1);
        for j=1:sample_count,
            ridx = randsample(I_cols,shreds);
            rvs(j) = mean(col_diffs(ridx));
        end
        sc_bs_diffs(i,:) = rvs(:);
    end
    % For each shred count, compute its probability under a Gamma distribution,
    % with parameters inferred from the bootstrap distributions
    for i=1:sc_count,
        p = gamfit(sc_bs_diffs(i,:));
        sc_gpdfs(i) = log(gampdf(sc_diffs(i),p(1),p(2)));
    end
    sc_gpdfs = [repmat(sc_gpdfs(1),5,1); sc_gpdfs; repmat(sc_gpdfs(end),5,1)];
    sc_gpdfs = conv(sc_gpdfs,normpdf(-5:5,0,1.5),'same');
    sc_gpdfs = sc_gpdfs(6:end-5);
    [min_score min_idx] = min(sc_gpdfs);
    opt_shred_count = shred_counts(min_idx) + 2;
end

% Compute the optimal shred count and its corresponding shred intervals
intervals = sort(cd_sort_idx(1:opt_shred_count),'ascend');
intervals = [1 intervals I_cols + 1]; 
shred_mask = [];
for i=2:numel(intervals),
    shred_mask = [shred_mask; (1:I_cols < intervals(i)) & (1:I_cols >= intervals(i-1))];
end
shred_mask = logical(shred_mask);

% For the selected shred boundaries, get pairwise L/R match probabilities
match_probs = zeros(size(shred_mask,1));
for i=1:size(shred_mask,1),
    for j=1:size(shred_mask,1),
        shred_left = Id(:,shred_mask(i,:),:);
        size_left = size(shred_left,2);
        shred_right = Id(:,shred_mask(j,:),:);
        diffs = cat(1,raw_diffs(shred_mask(i,:)),raw_diffs(shred_mask(j,:)));
        join_diff = squeeze(shred_right(:,1,:) - shred_left(:,end,:));
        diffs(size_left+1) = sum(sqrt(sum(join_diff.^2,2))) / size(shred_left,1);
        diffs(1) = mean(diffs(2:4));
        diffs = diffs ./ conv(diffs,normpdf(-7:7,0.0,2.0),'same');
        match_probs(i,j) = normpdf(diffs(size_left+1),cds_mu,cds_sigma);
    end
end

% Get an order for the shreds, based on the computed match probabilities
% Compute order using Viterbi algorithm, once from left to right and once from
% right to left.
order_ltr = order_shreds_viterbi(match_probs);
order_rtl = fliplr(order_shreds_viterbi(match_probs'));
final_order = [];

if (sum(abs(order_ltr - order_rtl)) > 0.5)
    % For the two predicted shred orders, for each shred boundary, try joining
    % the initial subset of ltr shreds to the final subset of rtl shreds, and
    % vise-versa, checking to see which join point produces minimal error.
    fprintf('*');
    ltr_repeats = numel(unique(order_ltr)) ~= numel(order_ltr);
    rtl_repeats = numel(unique(order_rtl)) ~= numel(order_rtl);
    if (ltr_repeats || rtl_repeats)
        % Use the non-repeating order if possible, when some order repeats
        if (~ltr_repeats)
            final_order = order_ltr;
        end
        if (~rtl_repeats)
            final_order = order_rtl;
        end
    end
    if (numel(final_order) < 2)
        % QUICK N DIRTY: CHECK EACH ORDER, KEEP BEST ONE
        I_ltr = [];
        for i=1:numel(order_ltr),
            I_ltr = cat(2,I_ltr,I_shred(:,shred_mask(order_ltr(i),:),:));
        end
        diff_ltr = 0;
        for i=2:size(I_ltr,2),
            diff_ltr = diff_ltr + sum(sum(sum((I_ltr(:,i,:)-I_ltr(:,i-1,:)).^2)));
        end
        I_rtl = [];
        for i=1:numel(order_rtl),
            I_rtl = cat(2,I_rtl,I_shred(:,shred_mask(order_rtl(i),:),:));
        end
        diff_rtl = 0;
        for i=2:size(I_rtl,2),
            diff_rtl = diff_rtl + sum(sum(sum((I_rtl(:,i,:)-I_rtl(:,i-1,:)).^2)));
        end
        if (diff_ltr <= diff_rtl)
            final_order = order_ltr;
        else
            final_order = order_rtl;
        end
    end
else
    % Orders ltr and rtl are identical, so no need to check join points
    final_order = order_ltr;
end

% Deshred image based on final computed shred boundaries and order
I = [];
for i=1:numel(final_order),
    I = cat(2,I,I_shred(:,shred_mask(final_order(i),:),:));
end

return

end

function [ shred_order ] = order_shreds_viterbi( match_probs )
% For the given match probabilities, compute a shred order
shred_count = size(match_probs,1);
log_probs = -log(match_probs);
log_probs = log_probs + 100*eye(size(log_probs));
prev_shreds = zeros(shred_count,shred_count);
prev_probs = zeros(shred_count,shred_count);
prev_paths = zeros(shred_count,shred_count);
prev_paths_temp = zeros(shred_count,shred_count);
% Init first column of prev_probs
prev_probs(:,1) = -min(log_probs)';
prev_paths(:,1) = (1:shred_count)';
% Fill medial columns of prev_probs and prev_shreds
for t=2:shred_count,
    for i=1:shred_count,
        path_probs = prev_probs(:,t-1) + log_probs(:,i);
%         for j=1:shred_count,
%             if ismember(i,prev_paths(j,1:t-1))
%                 path_probs(j) = path_probs(j) + 100;
%             end
%         end
        [max_prob max_idx] = min(path_probs);
        prev_probs(i,t) = max_prob;
        prev_shreds(i,t) = max_idx;
        prev_paths_temp(i,t) = i;
        prev_paths_temp(i,1:t-1) = prev_paths(max_idx,1:t-1);
    end
    prev_paths = prev_paths_temp(:,:);
end
% Determine final shred
end_probs = -min(match_probs,[],2) + prev_probs(:,shred_count);
[end_prob end_idx] = min(end_probs);
shred_order = prev_paths(end_idx,:);

return

end
