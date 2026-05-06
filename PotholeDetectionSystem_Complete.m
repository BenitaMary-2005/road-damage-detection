%% ============================================================
%  COMPREHENSIVE POTHOLE DETECTION SYSTEM - ROAD-FOCUS VERSION
%  Improved: ROI Masking, Adaptive Thresholding, Crack Detection
%% ============================================================
clear all; close all; clc;

fprintf('============================================================\n');
fprintf('     POTHOLE DETECTION SYSTEM - ROAD-FOCUS EDITION\n');
fprintf('============================================================\n\n');

%% ========== USER INPUTS & CONFIGURATION ==========
fprintf('Select Input Type:\n');
fprintf('  1. Single Image\n  2. Multiple Images\n  3. Video File\n');
inputType = input('Enter choice (1-3): ');

switch inputType
    case 1, mediaPath = input('Enter image path: ', 's');
    case 2, mediaPath = input('Enter folder path: ', 's');
    case 3, mediaPath = input('Enter video path: ', 's');
end

% Path Cleanup
mediaPath = strrep(mediaPath, '"', '');
mediaPath = strtrim(mediaPath);

fprintf('\nWeather Condition: 1. Dry | 2. Wet | 3. Night\n');
weatherChoice = input('Enter choice (1-3): ');
weatherMap = {'Dry', 'Wet', 'Night'};
weatherCondition = weatherMap{max(1, min(3, weatherChoice))};

gpsLat = input('Enter GPS Latitude: ');
gpsLon = input('Enter GPS Longitude: ');

%% ========== MAIN DISPATCH ==========
switch inputType
    case 1
        processImage(mediaPath, weatherCondition, gpsLat, gpsLon);
    case 2
        processBatchImages(mediaPath, weatherCondition, gpsLat, gpsLon);
    case 3
        processVideo(mediaPath, weatherCondition, gpsLat, gpsLon);
end

%% =============================================================
%%                   IMAGE PROCESSING PIPELINE
%% =============================================================
function processImage(imgPath, weatherCondition, gpsLat, gpsLon)
    if isfile(imgPath)
        img = imread(imgPath);
    else
        fprintf('File not found. Using demo image.\n');
        img = createDemoImage();
    end

    % 1. Preprocess with ROI Masking (Focus on the road)
    [imgProcessed, mask] = preprocessImage(img, weatherCondition);
    
    % 2. Detect based on texture and shape
    detections = heuristicPotholeDetection(imgProcessed, mask);
    
    % 3. Lane Analysis
    [lanes, ~] = detectLane(img);
    
    % 4. Analyze each detection
    for i = 1:length(detections)
        detections(i).lane = assignLane(detections(i).bbox, lanes);
        
        % Extract patch for severity calculation
        b = detections(i).bbox;
        r_range = floor(max(1, b(2)):min(size(img,1), b(2)+b(4)));
        c_range = floor(max(1, b(1)):min(size(img,2), b(1)+b(3)));
        patch = imgProcessed(r_range, c_range);
        
        detections(i).severity = calculateSeverity(patch, b, weatherCondition);
        detections(i).riskLevel = getRiskLevel(detections(i).severity);
    end
    
    % 5. Visualization
    imgAnnotated = annotateImage(img, detections, weatherCondition);
    
    figure('Name', 'Road Analysis Results', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);
    subplot(2,2,1); imshow(img); title('Original Input');
    subplot(2,2,2); imshow(imgProcessed); title('Road-Surface Isolation (ROI)');
    subplot(2,2,3); imshow(imgAnnotated); title('Detected Faults (Potholes/Cracks)');
    subplot(2,2,4); visualizeResults(img, detections); title('Severity Map');
end

%% =============================================================
%%                   CORE LOGIC FUNCTIONS
%% =============================================================

function [imgOut, mask] = preprocessImage(img, weather)
    [h, w, ~] = size(img);
    if size(img, 3) == 3, gray = rgb2gray(img); else, gray = img; end
    
    % --- ROAD ROI MASKING ---
    % Defines a trapezoid focusing on the bottom half/center of the image
    % Points: [Bottom-Left; Mid-Left; Mid-Right; Bottom-Right]
    roiPoints = [0.05*w, h; 0.4*w, 0.45*h; 0.6*w, 0.45*h; 0.95*w, h];
    mask = poly2mask(roiPoints(:,1), roiPoints(:,2), h, w);
    
    % Apply mask and enhance contrast
    gray(~mask) = 255; % Set non-road areas to white (ignore)
    imgOut = adapthisteq(gray, 'ClipLimit', 0.03);
end

function detections = heuristicPotholeDetection(img, mask)
    % Adaptive thresholding to find dark spots relative to road color
    bw = imbinarize(img, 'adaptive', 'Sensitivity', 0.45);
    bw = ~bw; % Invert: objects of interest are now white
    bw(~mask) = 0; % Ensure nothing outside the road is kept
    
    % Clean noise: remove tiny dots and bridge small gaps
    bw = imopen(bw, strel('disk', 2));
    bw = imclose(bw, strel('disk', 4));
    
    % Analyze shapes
    stats = regionprops(bw, 'BoundingBox', 'Area', 'Solidity', 'Eccentricity');
    
    detections = struct([]);
    count = 0;
    for i = 1:length(stats)
        % Criteria: Significant size, solid shape (not wispy)
        if stats(i).Area > 250 && stats(i).Solidity > 0.5
            count = count + 1;
            detections(count).bbox = stats(i).BoundingBox;
            
            % Distinguish between Potholes (roundish) and Cracks (long)
            if stats(i).Eccentricity > 0.92
                detections(count).type = 'Crack';
            else
                detections(count).type = 'Pothole';
            end
        end
    end
end

function imgOut = annotateImage(img, detections, weather)
    imgOut = img;
    for i = 1:length(detections)
        bbox = detections(i).bbox;
        if strcmp(detections(i).type, 'Pothole')
            color = 'red';
        else
            color = 'yellow'; % Cracks
        end
        
        imgOut = insertShape(imgOut, 'rectangle', bbox, 'LineWidth', 3, 'Color', color);
        label = sprintf('%s (Sev: %.0f)', detections(i).type, detections(i).severity);
        imgOut = insertText(imgOut, [bbox(1), bbox(2)-25], label, 'BoxColor', color, 'TextColor', 'black');
    end
end

function severity = calculateSeverity(patch, bbox, weather)
    if isempty(patch), severity = 0; return; end
    % Heuristic based on size and contrast
    sizeFactor = min(50, (bbox(3)*bbox(4))/800);
    contrastFactor = (1 - mean(patch(:))/255) * 50;
    severity = sizeFactor + contrastFactor;
    if strcmp(weather, 'Wet'), severity = severity * 1.2; end
    severity = min(100, severity);
end

function risk = getRiskLevel(sev)
    if sev > 70, risk = 'Critical'; elseif sev > 35, risk = 'Medium'; else, risk = 'Low'; end
end

%% =============================================================
%%                   STUB / HELPER FUNCTIONS
%% =============================================================
function [lanes, laneImg] = detectLane(img)
    w = size(img, 2);
    lanes(1).centerX = w*0.25; lanes(2).centerX = w*0.5; lanes(3).centerX = w*0.75;
    laneImg = img; 
end

function laneIdx = assignLane(bbox, lanes)
    cx = bbox(1) + bbox(3)/2;
    [~, laneIdx] = min(abs([lanes.centerX] - cx));
end

function visualizeResults(img, detections)
    imshow(img); hold on;
    for i = 1:length(detections)
        b = detections(i).bbox;
        plot(b(1)+b(3)/2, b(2)+b(4)/2, 'r*', 'MarkerSize', 10);
    end
end

function img = createDemoImage()
    img = uint8(repmat(linspace(120, 100, 480)', [1, 640, 3])); % Asphalt gradient
    img(300:330, 300:340, :) = 30; % Fake dark pothole
    img(350:380, 200:205, :) = 20; % Fake thin crack
end

function processBatchImages(folderPath, weather, lat, lon)
    files = dir(fullfile(folderPath, '*.jpg'));
    for f = 1:numel(files)
        processImage(fullfile(files(f).folder, files(f).name), weather, lat, lon);
    end
end

function processVideo(videoPath, weatherCondition, gpsLat, gpsLon)
    if ~isfile(videoPath), fprintf('Video not found.\n'); return; end
    v = VideoReader(videoPath);
    
    % Create a figure for playback
    fig = figure('Name', 'Real-Time Road Analysis', 'NumberTitle', 'off');
    
    while hasFrame(v) && ishghandle(fig)
        frame = readFrame(v);
        
        % 1. Preprocess (ROI Masking)
        [imgProcessed, mask] = preprocessImage(frame, weatherCondition);
        
        % 2. Detect
        detections = heuristicPotholeDetection(imgProcessed, mask);
        
        % 3. FIX: Calculate Severity for each detection (Prevents the error)
        for i = 1:length(detections)
            b = detections(i).bbox;
            
            % Crop the patch for analysis
            r_range = floor(max(1, b(2)):min(size(frame,1), b(2)+b(4)));
            c_range = floor(max(1, b(1)):min(size(frame,2), b(1)+b(3)));
            patch = imgProcessed(r_range, c_range);
            
            % Assign values to the struct
            detections(i).severity = calculateSeverity(patch, b, weatherCondition);
            detections(i).riskLevel = getRiskLevel(detections(i).severity);
        end
        
        % 4. Display
        imshow(annotateImage(frame, detections, weatherCondition));
        drawnow; 
    end
end