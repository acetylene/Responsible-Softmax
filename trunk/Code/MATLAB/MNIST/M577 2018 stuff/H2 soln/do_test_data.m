function [Y,T] = do_test_data(digit0, digit1, W, V, show_misclassified)
image_filename='t10k-images-idx3-ubyte';
labels_filename='t10k-labels-idx1-ubyte';
[I,T]=decode_images(image_filename,labels_filename);

idx=(T==digit0|T==digit1);
I=I(idx,:);
T=T(idx,:);

if ~isempty(V)
    M = I * V;
    Y=1./(1+exp(-(W(1)+W(2:end)*M')));
else
    Y=1./(1+exp(-(W(1)+W(2:end)*I')));
end
Y=Y';
T = T==digit0;

if show_misclassified
    % Display misclassified test images
    err_pos = find(T ~= round(Y));
    % Show error positions
    for j = 1:length(err_pos)
        l = err_pos(j);
        imagesc(reshape(I(l,:),[28,28])'), 
        if T(l) == 1
            digit = digit0;
        else
            digit = digit1;
        end
        title(['Misclassified digit: ',num2str(digit)]);
        drawnow;
        pause(1);
    end
end