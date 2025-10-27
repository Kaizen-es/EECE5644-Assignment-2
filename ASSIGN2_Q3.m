%{
                    EECE5644 FALL 2025 - ASSIGNMENT 2
                                QUESTION 3
                  BAYESIAN MAP VEHICLE LOCALIZATION
%}

clear all, close all,
fprintf('\nQUESTION 3: MAP VEHICLE LOCALIZATION\n');

% PARAMETERS
sigmaMeas = 0.3;     % Measurement noise std dev
sigmaX = 0.25;       % Prior std dev for x
sigmaY = 0.25;       % Prior std dev for y

% GENERATE TRUE VEHICLE POSITION
% Single random position for all K values - Adapted from feedback from Chatgpt
rng(73);
angleTrueVehicle = 2*pi*rand();
radiusTrueVehicle = 0.7*rand();
xTrue = radiusTrueVehicle * cos(angleTrueVehicle);
yTrue = radiusTrueVehicle * sin(angleTrueVehicle);
posTrue = [xTrue; yTrue];

fprintf('True vehicle position: (%.3f, %.3f)\n', xTrue, yTrue);

% GRID FOR CONTOUR EVALUATION
% Standard MATLAB grid evaluation approach - Adapted from discussion with Claude
gridPoints = linspace(-2, 2, 200);
[X, Y] = meshgrid(gridPoints, gridPoints);

% DETERMINE COMMON CONTOUR LEVELS
% Generate evenly-spaced landmarks and measurements for each K - Adapted from discussion with Claude
allObjectives = [];
for K = 1:4
    % Generate K evenly-spaced landmarks on unit circle
    angles = linspace(0, 2*pi, K+1);
    landmarks = [cos(angles(1:K)); sin(angles(1:K))];
    
    % Generate measurements for these specific landmarks
    measurements = generateMeasurements(posTrue, landmarks, sigmaMeas);
    
    % Evaluate objective
    Z = evaluateObjective(X, Y, measurements, landmarks, sigmaMeas, sigmaX, sigmaY);
    allObjectives = [allObjectives; Z(:)];
end
% Use 95th percentile to avoid outliers affecting contour range
contourLevels = linspace(min(allObjectives), prctile(allObjectives, 95), 15);

fprintf('Using %d common contour levels: [%.2f, %.2f]\n\n', ...
    length(contourLevels), min(contourLevels), max(contourLevels));

% EVALUATE AND PLOT FOR EACH K - Adapted from discussion with Claude
figure(1), clf;
for K = 1:4
    fprintf('K = %d landmarks:\n', K);
    
    % Generate K evenly-spaced landmarks on unit circle
    % Each K gets its own proper geometric configuration
    angles = linspace(0, 2*pi, K+1);
    landmarks = [cos(angles(1:K)); sin(angles(1:K))];
    
    fprintf('  Landmark positions:\n');
    for i = 1:K
        fprintf('    Landmark %d: (%.3f, %.3f) at angle %.1f°\n', ...
            i, landmarks(1,i), landmarks(2,i), angles(i)*180/pi);
    end
    
    % Generate measurements for these specific landmarks
    measurements = generateMeasurements(posTrue, landmarks, sigmaMeas);
    
    fprintf('  Measurements: ');
    fprintf('%.3f ', measurements);
    fprintf('\n');
    
    % Evaluate MAP objective over grid
    Z = evaluateObjective(X, Y, measurements, landmarks, sigmaMeas, sigmaX, sigmaY);
    
    % Find MAP estimate (minimum of objective)
    [minObj, minIdx] = min(Z(:));
    [minI, minJ] = ind2sub(size(Z), minIdx);
    xMAP = X(minI, minJ);
    yMAP = Y(minI, minJ);
    errorDist = norm([xMAP; yMAP] - posTrue);
    
    fprintf('  MAP estimate: (%.3f, %.3f)\n', xMAP, yMAP);
    fprintf('  Error from true: %.4f\n', errorDist);
    fprintf('  Objective value at MAP: %.3f\n\n', minObj);
    
    % Plot - Standard MATLAB contour plotting
    subplot(2, 2, K);
    contour(X, Y, Z, contourLevels, 'LineWidth', 1.5); hold on;
    
    % Mark true position with + per assignment specification
    plot(xTrue, yTrue, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
    
    % Mark landmarks with o per assignment specification
    plot(landmarks(1,:), landmarks(2,:), 'ko', 'MarkerSize', 10, ...
        'LineWidth', 2, 'MarkerFaceColor', 'k');
    
    % Mark MAP estimate for visualization
    plot(xMAP, yMAP, 'g*', 'MarkerSize', 12, 'LineWidth', 2);
    
    % Draw unit circle for reference
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1);
    
    axis equal;
    xlim([-2, 2]);
    ylim([-2, 2]);
    grid on;
    xlabel('x (horizontal coordinate)');
    ylabel('y (vertical coordinate)');
    title(sprintf('K = %d Landmarks', K));
    legend('Objective Contours', 'True Position', 'Landmarks', ...
        'MAP Estimate', 'Unit Circle', 'Location', 'best');
end

sgtitle('MAP Vehicle Localization: Effect of Number of Landmarks');

% DISCUSSION
fprintf(' OBSERVATIONS \n');
fprintf('Landmark Geometry:\n');
fprintf('  K=1: Single landmark at 0°\n');
fprintf('  K=2: Landmarks at 0°, 180° (diameter)\n');
fprintf('  K=3: Landmarks at 0°, 120°, 240° (equilateral triangle)\n');
fprintf('  K=4: Landmarks at 0°, 90°, 180°, 270° (square)\n\n');

fprintf('Contour Behavior:\n');
fprintf('  K=1: Elongated contours showing high directional uncertainty\n');
fprintf('       (vehicle constrained to ring around single landmark)\n');
fprintf('  K=2: Elliptical contours with two potential minima\n');
fprintf('       (ambiguity from two circle intersections)\n');
fprintf('  K=3: Tighter, more circular contours\n');
fprintf('       (triangulation resolves position uniquely)\n');
fprintf('  K=4: Very tight contours centered near true position\n');
fprintf('       (redundant measurements increase confidence)\n\n');

fprintf('MAP Estimate Accuracy:\n');
fprintf('  As K increases, MAP estimate approaches true position\n');
fprintf('  K>=3 provides sufficient geometric constraints for localization\n\n');

fprintf('Prior Effect:\n');
fprintf('  Prior term regularizes estimate toward origin\n');
fprintf('  Acts as "tiebreaker" when measurements are ambiguous (low K)\n');
fprintf('  Becomes less influential as K increases (more measurement data)\n');
fprintf('  As σₓ²,σᵧ² → ∞: prior term → 0, MAP → ML (unregularized)\n\n');


% HELPER FUNCTIONS
% Measurement generation with rejection sampling - Adapted from discussion with Claude
function measurements = generateMeasurements(truePos, landmarks, sigmaNoise)
    K = size(landmarks, 2);
    measurements = zeros(K, 1);
    
    for i = 1:K
        validMeasurement = false;
        while ~validMeasurement
            trueDist = norm(truePos - landmarks(:,i));
            noisyMeasurement = trueDist + sigmaNoise * randn();
            
            % Reject negative measurements per assignment specification
            if noisyMeasurement >= 0
                measurements(i) = noisyMeasurement;
                validMeasurement = true;
            end
        end
    end
end

% MAP objective function evaluation - Adapted from discussion with Claude
function Z = evaluateObjective(X, Y, measurements, landmarks, sigmaMeas, sigmaX, sigmaY)
    K = length(measurements);
    Z = zeros(size(X));
    
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            x = X(i,j);
            y = Y(i,j);
            
            % Likelihood term: (1/2σ²)Σ[rᵢ - dᵢ(x,y)]²
            % Represents -log p(r|x) from Gaussian measurement model
            likelihoodTerm = 0;
            for k = 1:K
                predictedDist = sqrt((x - landmarks(1,k))^2 + (y - landmarks(2,k))^2);
                residual = measurements(k) - predictedDist;
                likelihoodTerm = likelihoodTerm + residual^2;
            end
            likelihoodTerm = likelihoodTerm / (2 * sigmaMeas^2);
            
            % Prior term: (1/2)[x²/σₓ² + y²/σᵧ²]
            % Represents -log p(x) from zero-mean Gaussian prior
            priorTerm = 0.5 * ((x^2 / sigmaX^2) + (y^2 / sigmaY^2));
            
            % Total MAP objective (to be minimized)
            Z(i,j) = likelihoodTerm + priorTerm;
        end
    end
end