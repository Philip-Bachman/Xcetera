% Test the image deshredding algorithm
I_true = imread('TokyoPanorama.png');
shred_count = 20;
min_shred = 20;
test_count = 200;

im_fig = figure();
errors = zeros(test_count,1);
for i=1:test_count,
    fprintf('.');
    I_shred = shred_im(I_true, shred_count, min_shred);
    I = deshred_new(I_shred, shred_count+2);
    if (numel(I_true) == numel(I))
        errors(i) = sum((I_true(:) - I(:)).^2);
    else
        errors(i) = sum(I_true(:).^2);
    end
    figure(im_fig);
    image(I);
    title(sprintf('Round %d',i));
    drawnow();
end
fprintf('\n');