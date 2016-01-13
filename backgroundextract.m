close all;
clear all;
clc;
        
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 300);
        
        obj.cornerDetector = vision.CornerDetector( ... 
             'Method', 'Local intensity comparison (Rosten & Drummond)');

videoinput = 'Vehicle Counter.mp4';
readobj = VideoReader(videoinput);
nFrames = readobj.NumberOfFrames;
videoframe = zeros(readobj.Height, readobj.Width);
int_region = zeros(readobj.Height, readobj.Width);
int_region(158:306, :) = 1;

        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'corners_vector', {},...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});

FF = 500;
load('Background_VC.mat');
for f = 3090:1:3091
    frame = im2double(rgb2gray(read(readobj,f)));
%     figure;imshow(frame);
    mask = abs(frame-Background_VC);
%     figure;imshow(mask);
    mask = mask.* int_region;
    mask = im2bw(uint8(255*mask),0.2);
    figure;
    imshow(mask,[]);
%     figure;imshow(mask);
    mask = imopen(mask, strel('rectangle', [3,3]));
        figure;
    imshow(mask,[]);
    mask = imclose(mask, strel('rectangle', [19, 19])); 
        figure;
    imshow(mask,[]);
    mask = imfill(mask, 'holes');
        figure;
    imshow(mask,[]);
%     figure;imshow(mask);

    [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
        
    mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
    for ii = 1:size(bboxes,1)
        obj.cornerDetector = vision.CornerDetector( ... 
             'Method', 'Local intensity comparison (Rosten & Drummond)');
         tmp = zeros(size(frame));
         tmp(bboxes(ii,2):bboxes(ii,2)+bboxes(ii,4),bboxes(ii,1):bboxes(ii,1)+bboxes(ii,3)) = 1;
        corners_pts{ii} = step(obj.cornerDetector, im2single(frame.*tmp));
        release(obj.cornerDetector);
    end
    %% Predict New Locations of Existing Tracks
    for i = 1:length(tracks)
        bbox = tracks(i).bbox;
        predictedCentroid = predict(kalmanFilter);
        predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
        tracks(i).bbox = [predictedCentroid, bbox(3:4)];
    end
    %% Assign Detections to Tracks
   
    
    %% display
    for ii = 1:size(bboxes,1)
        frame = insertObjectAnnotation(frame, 'rectangle', bboxes(ii,:), num2str(ii));
        frame = insertMarker(frame, corners_pts{ii}, 'Color', 'red');


        mask = insertObjectAnnotation(mask, 'rectangle', bboxes(ii,:), num2str(ii));
        mask = insertMarker(mask, corners_pts{ii}, 'Color', 'red');

    end
    
    figure;
    imshow(frame);    
    figure;
    imshow(mask);
end

% a1 = double([6501 20759]);
% a2 = double([7619 5848]);
% a3 = double([8654 4641]);
% 
% b1 = double([905+3224 3641+15924]);
% b2 = double([6706 5356]);
% b3 = double([9203 5215]);
% 
% d11 = norm(a1-b1);
% d12 = norm(a1-b2);
% d13 = norm(a1-b3);
% d21 = norm(a2-b1);
% d22 = norm(a2-b2);
% d23 = norm(a2-b3);
% d31 = norm(a3-b1);
% d32 = norm(a3-b2);
% d33 = norm(a3-b3);
% frames = zeros(readobj.Height, readobj.Width, FF);
% for f = 1:FF
%     frames (:,:,f) = im2double(rgb2gray(read(readobj,f)));
% end
% 
% Background_VC = median(frames,3);
% figure;imshow(Background_VC);