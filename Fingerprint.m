clc; clear; close all;

%% Step 1: Create Timestamped Output Folder
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
outputDir = ['fingerprint_outputs_' timestamp];
mkdir(outputDir);

imgName = '101_2.tif';  % Your fingerprint image

%% Step 2: Load and Preprocess Image
orig_img = imread(imgName);
img = imresize(orig_img, [512 512]);

if size(img, 3) == 3
    img = rgb2gray(img);
end

img = medfilt2(img, [3 3]);  % Slight smoothing
img = imsharpen(img, 'Radius', 1, 'Amount', 0.7);  % Mild sharpening

% Save comparison
f_compare = figure;
subplot(1,2,1), imshow(orig_img), title('Original');
subplot(1,2,2), imshow(img), title('Sharpened');
saveas(f_compare, fullfile(outputDir, '00_comparison_sharpened.png'));

f1 = figure; imshow(img); title('Grayscale + Sharpened');
saveas(f1, fullfile(outputDir, '01_sharpened.png'));

%% Step 3: Segment Fingerprint
mask = imbinarize(img, 'adaptive', ...
    'ForegroundPolarity', 'dark', 'Sensitivity', 0.45);
mask = imclose(mask, strel('disk', 3));
mask = imfill(mask, 'holes');

segmented = img;
segmented(~mask) = 255;

f2 = figure; imshow(segmented); title('Segmented Fingerprint');
saveas(f2, fullfile(outputDir, '02_segmented.png'));

%% Step 4: Orientation Field
[orientim, ~] = ridgeorient(segmented, 1, 3, 3);
spacing = 16;
[rows, cols] = size(segmented);
[X, Y] = meshgrid(1:spacing:cols, 1:spacing:rows);
orient_quiver = orientim(1:spacing:end, 1:spacing:end);
U = cos(orient_quiver);
V = -sin(orient_quiver);  % Negative for image coordinates

f3 = figure; imshow(segmented); hold on;
quiver(X, Y, U, V, 0.5, 'r');
title('Ridge Orientation (Flow)');
saveas(f3, fullfile(outputDir, '03_orientation.png'));

%% Step 5: Level 1 Features - Core, Delta, Pattern Type
singularMap = detect_singularities(orientim);
[core_y, core_x] = find(singularMap == 1);
[delta_y, delta_x] = find(singularMap == -1);

f4 = figure; imshow(segmented); hold on;
plot(core_x, core_y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(delta_x, delta_y, 'go', 'MarkerSize', 10, 'LineWidth', 2);
legend('Core', 'Delta');
title('Level 1: Core (Red), Delta (Green)');
saveas(f4, fullfile(outputDir, '04_level1_features.png'));

% Save CSV
writematrix([core_x core_y], fullfile(outputDir, 'core_points.csv'));
writematrix([delta_x delta_y], fullfile(outputDir, 'delta_points.csv'));

% Pattern Classification
num_cores = numel(core_x);
num_deltas = numel(delta_x);
if num_cores == 1 && num_deltas == 1
    pattern = 'Loop';
elseif num_cores >= 1 && num_deltas >= 2
    pattern = 'Whorl';
elseif num_cores == 0 && num_deltas == 0
    pattern = 'Arch';
else
    pattern = 'Complex or Noisy';
end

disp(['Detected Pattern Type: ', pattern]);
fid = fopen(fullfile(outputDir, 'pattern_type.txt'), 'w');
fprintf(fid, 'Detected Pattern: %s\n', pattern);
fclose(fid);

%% Step 6: Level 2 - Minutiae Detection
bw = imbinarize(segmented);
thin_img = bwmorph(~bw, 'thin', Inf);

[min_end, min_bif] = extract_minutiae(thin_img);

f5 = figure; imshow(~thin_img); hold on;
plot(min_end(:,2), min_end(:,1), 'ro');
plot(min_bif(:,2), min_bif(:,1), 'go');
legend('Ridge Ending', 'Bifurcation');
title('Level 2: Minutiae');
saveas(f5, fullfile(outputDir, '05_level2_minutiae.png'));

writematrix(min_end, fullfile(outputDir, 'ridge_endings.csv'));
writematrix(min_bif, fullfile(outputDir, 'bifurcations.csv'));

disp(['✔ All outputs saved to folder: ', outputDir]);

%% ----------------------- HELPER FUNCTIONS -----------------------

function [ridge_end, bifurcation] = extract_minutiae(thin)
    [rows, cols] = size(thin);
    ridge_end = [];
    bifurcation = [];

    for i = 2:rows-1
        for j = 2:cols-1
            if thin(i,j)
                n = thin(i-1:i+1, j-1:j+1);
                cn = sum(n(:)) - 1;
                if cn == 1
                    ridge_end = [ridge_end; i j];
                elseif cn >= 3
                    bifurcation = [bifurcation; i j];
                end
            end
        end
    end
end

function singularMap = detect_singularities(orient)
    [rows, cols] = size(orient);
    singularMap = zeros(rows, cols);
    orient_deg = orient * 180 / pi;

    for i = 2:rows-1
        for j = 2:cols-1
            block = orient_deg(i-1:i+1, j-1:j+1);
            angles = block([1 2 3 6 9 8 7 4 1]);
            diff_angle = angles(2:end) - angles(1:end-1);
            diff_angle = mod(diff_angle + 180, 360) - 180;
            sum_angle = sum(diff_angle);

            if abs(sum_angle - 180) < 30
                singularMap(i,j) = 1;   % Core
            elseif abs(sum_angle + 180) < 30
                singularMap(i,j) = -1;  % Delta
            end
        end
    end
end

function [orientim, reliability] = ridgeorient(im, gradientsigma, blocksigma, orientsmoothsigma)
    im = double(im);
    sze = fix(6 * gradientsigma); if mod(sze,2)==0, sze = sze+1; end
    gauss = fspecial('gaussian', sze, gradientsigma);
    [fx, fy] = gradient(gauss);
    Gx = filter2(fx, im); Gy = filter2(fy, im);
    Gxx = Gx.^2; Gyy = Gy.^2; Gxy = Gx.*Gy;

    sze = fix(6 * blocksigma); if mod(sze,2)==0, sze = sze+1; end
    gauss = fspecial('gaussian', sze, blocksigma);
    Gxx = filter2(gauss, Gxx); Gyy = filter2(gauss, Gyy);
    Gxy = 2 * filter2(gauss, Gxy);

    denom = sqrt(Gxy.^2 + (Gxx - Gyy).^2) + eps;
    sin2theta = Gxy ./ denom; cos2theta = (Gxx - Gyy) ./ denom;

    sze = fix(6 * orientsmoothsigma); if mod(sze,2)==0, sze = sze+1; end
    gauss = fspecial('gaussian', sze, orientsmoothsigma);
    sin2theta = filter2(gauss, sin2theta);
    cos2theta = filter2(gauss, cos2theta);

    orientim = 0.5 * atan2(sin2theta, cos2theta);
    reliability = denom ./ max(Gxx + Gyy, eps);
end
