function [f, count, Vehicle_buff, frame_buff, vector_buff, pts_buff] = tracking_classification(Background_VC,videopath)
obj = setupEnvir();
tracks = init_Tracks(); 
nextId = 1; 
f = 0;
ID_buff = [];
Vehicle_buff = {};
frame_buff = {};
vector_buff = {};
pts_buff = {};
count = 0;
corners_pts = {};
while ~isDone(obj.reader)
    f = f+1;
    frame = readFrame();
%     frame1 = abs(rgb2gray(frame) - single(background_new/255));
%     frame1 = im2uint8(frame1);
%     frame1 = single((frame1/8) * 8)/255;
%     frame1 = frame1(1:260,1:338); %zone1
%     mask1 = zeros(size(frame1));
%     mask1(287:360,1:276) = 1;
%     frame1 = frame1;%.*mask1; 
%     frame1 = frame1(287:338,1:276); %zone2
%     frame1 = frame1(270:394,1:286); %zone2
    [corners_pts, corners_vector, centroids, bboxes, mask] = find_Objects(frame);
    predict_new_Tracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        Assignment_Track();
    
    update_Assigned_Tracks();
    update_Unassigned_Tracks();
    delete_Tracks();
    create_New_Tracks();
    
    %%%%%%%%%%%%%zone2%%%%%%%%%%%%%%%%%%%%%%%%
    % 213 gives 30 vehicles
    if ~isempty(tracks)
        for ii = 1:length(tracks);
            if tracks(ii).bbox(2)<=180 && (tracks(ii).bbox(2)+tracks(ii).bbox(4))>=210%||tracks(ii).bbox(2)==1 || (tracks(ii).bbox(2)+tracks(ii).bbox(4))==72
                if isempty(find(ID_buff==tracks(ii).id))
                    ID_buff(count+1) = tracks(ii).id;
                    Vehicle_buff{end+1} = frame(tracks(ii).bbox(2):tracks(ii).bbox(2)+tracks(ii).bbox(4),tracks(ii).bbox(1):tracks(ii).bbox(1)+tracks(ii).bbox(3));
                    frame_buff{end+1} = frame;
                    vector_buff{end+1} = tracks(ii).corners_vector;
                    pts_buff{end+1} = tracks(ii).corners_pts;
                    count = count+1;
                end
            end
        end
    end
    %%%%%%%%%%%%%zone2%%%%%%%%%%%%%%%%%%%%%%%%%
    display_Results();
end


    function obj = setupEnvir()
        obj.reader = vision.VideoFileReader(videopath);
        
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        
        obj.detector = vision.ForegroundDetector('NumGaussians', 5, ...
            'NumTrainingFrames', 400, 'MinimumBackgroundRatio', 0.7);
        
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 700);
    end



    function tracks = init_Tracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'corners_vector', {},...
            'corners_pts', {},...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end


    function frame = readFrame()
        frame = obj.reader.step();
    end


%% find detection using background subtraction and mophological operation
% find the location of corners on the vehicles 
    function [corners_pts, corners_vector, centroids, bboxes, mask] = find_Objects(frame)
        
        gray_frame = im2double(rgb2gray(frame));
        int_region = zeros(size(gray_frame));
        int_region(158:306, :) = 1;
        mask = abs(gray_frame-Background_VC);
        mask = mask.* int_region;
        mask = im2bw(uint8(255*mask),0.2);
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [17, 17])); 
%         mask = imopen(mask, strel('rectangle', [3,3]));
%         mask = imclose(mask, strel('rectangle', [17, 17])); 
        mask = imfill(mask, 'holes');

        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
        
        corners_pts = {};
        for i = 1:size(bboxes,1)
            obj.cornerDetector = vision.CornerDetector( ... 
                    'Method', 'Local intensity comparison (Rosten & Drummond)');
            tmp = zeros(size(gray_frame));
            tmp(bboxes(i,2):bboxes(i,2)+bboxes(i,4),bboxes(i,1):bboxes(i,1)+bboxes(i,3)) = 1;
            corners_pts{i} = step(obj.cornerDetector, im2single(gray_frame.*tmp));
            release(obj.cornerDetector);
        end
        
        corners_vector = [];
        for i = 1:size(centroids,1)
            %corners_vector(i,:) = sum(corners_pts{1,i});
            corners_vector(i,:) = sum(corners_pts{1,i})-size(corners_pts{1,i},1)*centroids(i,:);
        end
    end

%% predict new location of tracks based on kalmanfilter
    function predict_new_Tracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            predictedCentroid = predict(tracks(i).kalmanFilter);
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

%% assgn detections to tracks based on Hungarian algorithm

    function [assignments, unassignedTracks, unassignedDetections] = ...
            Assignment_Track()
        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        assignments = [];
        unassignedTracks = [];
        unassignedDetections = [];
        
        cost = zeros(nTracks, nDetections);
        cost1 = zeros(nTracks, nDetections);
        cost2 = zeros(nTracks, nDetections);
        for i = 1:nTracks
            %cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
            for j = 1:nDetections
                cost1(i, j) = norm(tracks(i).corners_vector-corners_vector(j,:))/sqrt(sum(tracks(i).corners_vector.^2 + corners_vector(j,:).^2));
            end
        end
        
        for i = 1:nTracks
            cost2(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        cost = im2double(cost1) + im2double(cost2);
        costOfNonAssignment = 100; %5
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);

 %% try to delete unreasonable detections 
 % if the detection's bounding box is include in the predicted bounding
 % box and if it has not been assigned to a existed track, it should be
 % deleted 
        detections_ind = [];
        noise_detections_ind = [];
        noise_tracks_ind = [];
        for i = 1:numel(unassignedDetections)
            bbox = bboxes(i,:);
            for j = 1:nTracks
                if  (1.4*bbox(2)>=tracks(j).bbox(2)) && (bbox(1)+bbox(3)<=tracks(j).bbox(1)+1.4*tracks(j).bbox(3)) &&...
                        (bbox(1)>=tracks(j).bbox(1)) && (bbox(2)+bbox(4)<=tracks(j).bbox(2)+tracks(j).bbox(4))
%                     (bbox(1)>=tracks(j).bbox(1) && bbox(1)+bbox(3)<=tracks(j).bbox(1)+1.5*tracks(j).bbox(3)) &&...
%                         (bbox(2)>=tracks(j).bbox(2) && bbox(2)+bbox(4)<=tracks(j).bbox(2)+1.5*tracks(j).bbox(4))
                    %bbox(1)>=tracks(j).bbox(1) && bbox(1)+bbox(3)<=tracks(j).bbox(1)+tracks(j).bbox(3) &&...
                    noise_detections_ind (end+1) = i;
                    noise_tracks_ind(end+1) = j;
                else
                   detections_ind (end+1) = i;
                end
            end
        end
        if size(detections_ind,2)<numel(unassignedDetections) && nTracks~=0
            unassignedDetections = unassignedDetections(detections_ind,:);
            for i = 1:numel(noise_detections_ind)
                target_detection_ind = assignments(find(assignments(:,1)==noise_tracks_ind(i)),2);
                target_bbox = bboxes(target_detection_ind,:);
                combined_bbox = bboxes(noise_detections_ind(i),:);
                min_x = min(target_bbox(1),combined_bbox(1));
                min_y = min(target_bbox(2),combined_bbox(2));
                max_x = max((target_bbox(1)+target_bbox(3)),(combined_bbox(1)+combined_bbox(3)));
                max_y = max((target_bbox(2)+target_bbox(4)),(combined_bbox(2)+combined_bbox(4)));
                bboxes(target_detection_ind,:) = [min_x, min_y, (max_x-min_x), (max_y-min_y)];
            end
        end
%         for i = 1:nTracks
%             min_cost = min(cost(i,:));
%             if min_cost > 5000
%                 unassignedTracks = [unassignedTracks; i];
%             else 
%                 assignments= [assignments; [i, find(cost(i,:)==min_cost)]];
%             end
% 
%         end
%         
%         if size(assignments, 1) < nDetections
%             if ~isempty(assignments)
%                 matched_ass = assignments(:, 2);
%                 alldetect = [1:nDetections]';
%                 for i = 1:nDetections
%                     match_num = find(matched_ass == alldetect(i));
%                     if isempty(match_num)
%                         unassignedDetections = [unassignedDetections; i];
%                     end
%                 end
%             else
%                 unassignedDetections = [1:nDetections]';
%             end
%         end

    end

%% update assigened tracks
    function update_Assigned_Tracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            correct(tracks(trackIdx).kalmanFilter, centroid);
            tracks(trackIdx).corners_vector = corners_vector(detectionIdx, :);
            %tracks(trackIdx).corners_pts = {corners_pts{1, detectionIdx}};
            %change cell 
            tracks(trackIdx).corners_pts = corners_pts{1, detectionIdx};
            
            tracks(trackIdx).bbox = bbox;
            
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

%% update unassigned tracks
    function update_Unassigned_Tracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% delete tracks
    function delete_Tracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 3;
        ageThreshold = 5;
        
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        tracks = tracks(~lostInds);
    end

%% crreat new tracks
    function create_New_Tracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        corners_vector = corners_vector(unassignedDetections, :);
        
        %corners_pts = {corners_pts{1,unassignedDetections'}};
        %transposition doesn't change anything
        corners_pts = {corners_pts{1,unassignedDetections'}};
        
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            cornervector = corners_vector(i, :);
            cornerpts = corners_pts{1,i};
            
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'corners_vector', cornervector,...
                'corners_pts', cornerpts,...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            tracks(end + 1) = newTrack;
            nextId = nextId + 1;
        end
    end

%% display result
    function display_Results()
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        
        minVisibleCount = 4;
        if ~isempty(tracks)
              
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            if ~isempty(reliableTracks)
                bboxes = cat(1, reliableTracks.bbox);
                
                ids = int32([reliableTracks(:).id]);

                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end
        
        obj.maskPlayer.step(mask);        
        obj.videoPlayer.step(frame);
    end
displayEndOfDemoMessage(mfilename)
end