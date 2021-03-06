function [ I_shred ] = shred_im( I, shred_count, min_width )
% Vertically shred the image in I into shred_count shreds, each having width
% greater than, or equal to min_width. I is 3d (row x column x color).

cols = size(I,2);

edges = zeros(shred_count+1,1);
edges(1) = 1;
edges(shred_count+1) = cols + 1;
for i=2:shred_count,
    free_edges = setdiff(1:cols,edges);
    min_w = 0;
    while (min_w < min_width)
        new_edge = randsample(free_edges,1);
        min_w = min(abs(edges - new_edge));
    end
    edges(i) = new_edge;
end
edges = sort(edges,'ascend');

mask = zeros(numel(edges)-1,cols);
for i=2:numel(edges),
    mask(i-1,:) = (1:cols < edges(i)) & (1:cols >= edges(i-1));
end
mask = logical(mask);

% Arrange shreds into randomized order
I_shred = [];
shred_order = randperm(size(mask,1));
for i=1:numel(shred_order),
    I_shred = cat(2,I_shred,I(:,mask(shred_order(i),:),:));
end

return

end

