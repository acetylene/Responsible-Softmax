function [I,T]=decode_images(image_filename, labels_filename)
default_image_filename='train-images-idx3-ubyte';
default_labels_filename='train-labels-idx1-ubyte';
if(nargin < 1)
  image_filename = default_image_filename;
end
if(nargin < 2)
  labels_filename=default_labels_filename;
end



[I,n_im] = decode_image_file(image_filename);
[T,n_lab] = decode_label_file(labels_filename);
if n_im ~= n_lab
  error('Number of images does not match the number of labels');
end
  
end


function [I,number_of_images] = decode_image_file(image_filename)

% File is big-endian
%
% [offset] [type]          [value]          [description]
% 0000     32 bit integer  0x00000803(2051) magic number
% 0004     32 bit integer  60000            number of images
% 0008     32 bit integer  28               number of rows
% 0012     32 bit integer  28               number of columns
% 0016     unsigned byte   ??               pixel
% 0017     unsigned byte   ??               pixel
% ........
% xxxx     unsigned byte   ??               pixel


fh = fopen(image_filename,'rb');

magic_number = fread(fh, 1, 'uint32','ieee-be');
if magic_number ~= 2051
  error('Incorrect magic number');
end

number_of_images = fread(fh, 1, 'uint32','ieee-be');

disp(['Number of images == ',num2str(number_of_images)]);

number_of_rows = fread(fh, 1, 'uint32','ieee-be');
if number_of_rows ~= 28
  error('Incorrect number of rows');
end

number_of_columns = fread(fh, 1, 'uint32','ieee-be');
if number_of_columns ~= 28
  error('Incorrect number of columns');
end

I=zeros([number_of_images,number_of_columns,number_of_rows]);

for j=1:number_of_images
  I(j,:,:) = fread(fh, [number_of_columns, number_of_rows], 'uint8');
end
  
fclose(fh);

end

function [T,number_of_labels] = decode_label_file(labels_filename)

% [offset] [type]          [value]          [description]
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
% 0004     32 bit integer  60000            number of items
% 0008     unsigned byte   ??               label
% 0009     unsigned byte   ??               label
% ........
% xxxx     unsigned byte   ??               label

% The labels values are 0 to 9. 

fh = fopen(labels_filename,'rb');

magic_number = fread(fh, 1, 'uint32','ieee-be');
if magic_number ~= 2049
  error('Incorrect magic number');
end

number_of_labels = fread(fh, 1, 'uint32','ieee-be');
disp(['Number of labels == ', num2str(number_of_labels)]);;


T=fread(fh, number_of_labels, 'uint8');


fclose(fh);

end