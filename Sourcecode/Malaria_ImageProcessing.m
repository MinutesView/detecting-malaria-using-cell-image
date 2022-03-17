function Iout = Malaria_ImageProcessing(filename,imageSize)

I = imread(filename);
Iout = imresize(I, [imageSize imageSize]);

end