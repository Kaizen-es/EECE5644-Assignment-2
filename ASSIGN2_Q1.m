%{
                    EECE5644 FALL 2025 - ASSIGNMENT 2
                                QUESTION 1 
                PARTS 1 & 2: CLASSIFICATION COMPARISON

NOTE ON COLLABORATION:
This implementation builds upon Professor's provided code:
- Professor's evalGaussian.m (included as helper function - used verbatim)
- Professor's generateDataFromGMM.m (included as helper function - used verbatim)
- Professor's ExpectedRiskMinimization.m (classification pattern adapted)
- Professor's LDAwithROCcurve.m (ROC generation pattern adapted)
- My ASSIGN1_Q1.m for ROCcurve function (originally adapted from Professor's LDAwithROCcurve.m)

Additional implementation work developed through collaboration with Claude AI:
- Logistic-linear model training (gradient descent pattern from Professor's gradientDescent_binaryCrossEntropy.m)
- Logistic-quadratic model training (gradient descent pattern from Professor's gradientDescent_binaryCrossEntropy.m)
- Model evaluation functions (pattern from Professor's logisticGeneralizedLinearModel.m)
- Integration of all components
- Visualization and analysis code

All implementation decisions, code integration, parameter selection, and final 
analysis represent my own work synthesized from these resources and course materials.
%}

clear all, close all,

fprintf('\nQUESTION 1\n');


% DATA GENERATION
n = 2;
P_L0 = 0.6;
P_L1 = 0.4;

% Class 0: 2-component GMM
m01 = [-0.9; -1.1]; 
m02 = [0.8; 0.75];
C01 = [0.75 0; 0 1.25];
C02 = [0.75 0; 0 1.25];
w01 = 0.5; w02 = 0.5;

% Class 1: 2-component GMM
m11 = [-1.1; 0.9];
m12 = [0.9; -0.75];
C11 = [0.75 0; 0 1.25];
C12 = [0.75 0; 0 1.25];
w11 = 0.5; w12 = 0.5;

fprintf('Data Distribution:\n');
fprintf('  P(L=0) = %.2f, P(L=1) = %.2f\n', P_L0, P_L1);
fprintf('  Each class: 2-component GMM with equal weights\n\n');

% Setup GMM parameters - Based on sample code (generateDataFromGMM.m)
gmmParameters.priors = [P_L0*w01, P_L0*w02, P_L1*w11, P_L1*w12];
gmmParameters.meanVectors = [m01, m02, m11, m12];
gmmParameters.covMatrices = cat(3, C01, C02, C11, C12);

% Generate datasets - Based on sample code (generateDataFromGMM.m)
[xTrain50, componentLabels50] = generateDataFromGMM(50, gmmParameters);
[xTrain500, componentLabels500] = generateDataFromGMM(500, gmmParameters);
[xTrain5000, componentLabels5000] = generateDataFromGMM(5000, gmmParameters);

N = 10000;
[xVal, componentLabelsVal] = generateDataFromGMM(N, gmmParameters);

% Convert component labels to class labels
labelsTrain50 = double(componentLabels50 > 2);
labelsTrain500 = double(componentLabels500 > 2);
labelsTrain5000 = double(componentLabels5000 > 2);
labelsVal = double(componentLabelsVal > 2);

fprintf('  D_train^50:   %d samples (Class 0: %d, Class 1: %d)\n', ...
    50, sum(labelsTrain50==0), sum(labelsTrain50==1));
fprintf('  D_train^500:  %d samples (Class 0: %d, Class 1: %d)\n', ...
    500, sum(labelsTrain500==0), sum(labelsTrain500==1));
fprintf('  D_train^5000: %d samples (Class 0: %d, Class 1: %d)\n', ...
    5000, sum(labelsTrain5000==0), sum(labelsTrain5000==1));
fprintf('  D_validate:   %d samples (Class 0: %d, Class 1: %d)\n\n', ...
    N, sum(labelsVal==0), sum(labelsVal==1));

Nc = [sum(labelsVal==0), sum(labelsVal==1)];

% Visualize validation data
figure(1), clf,
plot(xVal(1,labelsVal==0), xVal(2,labelsVal==0), 'ob'); hold on;
plot(xVal(1,labelsVal==1), xVal(2,labelsVal==1), '+r');
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title('Validation Data: 2-Component GMM per Class');
legend('Class 0', 'Class 1', 'Location', 'best');

% PART 1: THEORETICALLY OPTIMAL CLASSIFIER

fprintf('\nPART 1: THEORETICALLY OPTIMAL CLASSIFIER \n');

% Theoretical gamma - Based on sample code (ExpectedRiskMinimization.m)
gamma_theoretical = P_L0 / P_L1;
fprintf('Theoretical gamma = P(L=0)/P(L=1) = %.4f\n\n', gamma_theoretical);

% Compute class-conditional likelihoods - Based on sample code (ExpectedRiskMinimization.m)
p_x_given_L0 = zeros(1, N);
p_x_given_L1 = zeros(1, N);

for i = 1:N
    p_x_given_L0(i) = w01*evalGaussian(xVal(:,i), m01, C01) + ...
                      w02*evalGaussian(xVal(:,i), m02, C02);
    p_x_given_L1(i) = w11*evalGaussian(xVal(:,i), m11, C11) + ...
                      w12*evalGaussian(xVal(:,i), m12, C12);
end

% Discriminant score - Based on sample code (ExpectedRiskMinimization.m)
discriminantScore = log(p_x_given_L1 ./ p_x_given_L0);

% Classification at theoretical gamma - Based on sample code (ExpectedRiskMinimization.m)
decision_theoretical = (discriminantScore >= log(gamma_theoretical));

% Confusion matrix - Based on sample code (ExpectedRiskMinimization.m)
ind00 = find(decision_theoretical==0 & labelsVal==0); p00 = length(ind00)/Nc(1);
ind10 = find(decision_theoretical==1 & labelsVal==0); p10 = length(ind10)/Nc(1);
ind01 = find(decision_theoretical==0 & labelsVal==1); p01 = length(ind01)/Nc(2);
ind11 = find(decision_theoretical==1 & labelsVal==1); p11 = length(ind11)/Nc(2);

Perror_theoretical = p10*P_L0 + p01*P_L1;

fprintf('Classification at Theoretical Gamma:\n');
fprintf('  P(error) = %.4f (%.2f%%)\n', Perror_theoretical, 100*Perror_theoretical);
fprintf('  TPR = %.4f, FPR = %.4f\n\n', p11, p10);

% Visualization - Based on sample code (ExpectedRiskMinimization.m)
figure(2), clf,
plot(xVal(1,ind00), xVal(2,ind00), 'og', 'MarkerSize', 4); hold on,
plot(xVal(1,ind10), xVal(2,ind10), 'or', 'MarkerSize', 6);
plot(xVal(1,ind01), xVal(2,ind01), '+r', 'MarkerSize', 6);
plot(xVal(1,ind11), xVal(2,ind11), '+g', 'MarkerSize', 4);
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title(sprintf('Part 1: Classification at Theoretical \\gamma = %.4f', gamma_theoretical));
legend('TN', 'FP', 'FN', 'TP');

% Generate ROC curve - Based on my ASSIGN1_Q1.m (originally from sample code LDAwithROCcurve.m)
[Pfp, Ptp, Pfn, Perror, thresholdList] = ROCcurve(discriminantScore, labelsVal, [P_L0, P_L1]);

% Find empirically optimal gamma - Adapted from discussion with Claude
[min_Perror, min_idx] = min(Perror);
empirical_gamma = exp(thresholdList(min_idx));

fprintf('Empirically Optimal Gamma:\n');
fprintf('  gamma* = %.4f\n', empirical_gamma);
fprintf('  Min P(error) = %.4f (%.2f%%)\n\n', min_Perror, 100*min_Perror);

% Plot ROC curve - Utilized suggestions from Copilot
figure(3), clf;
plot(Pfp, Ptp, 'b-', 'LineWidth', 2); hold on;
plot(Pfp(min_idx), Ptp(min_idx), 'go', 'MarkerSize', 12, 'LineWidth', 3);

[~, theory_idx] = min(abs(thresholdList - log(gamma_theoretical)));
plot(Pfp(theory_idx), Ptp(theory_idx), 'rs', 'MarkerSize', 12, 'LineWidth', 2);

xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('Part 1: ROC Curve - Theoretically Optimal Classifier');
legend('ROC Curve', ...
       sprintf('Empirical \\gamma*=%.3f, P(error)=%.4f', empirical_gamma, min_Perror), ...
       sprintf('Theoretical \\gamma=%.3f, P(error)=%.4f', gamma_theoretical, Perror_theoretical), ...
       'Location', 'southeast');
grid on;

% Decision boundary visualization - Based on sample code (ExpectedRiskMinimization.m)
x1_range = linspace(min(xVal(1,:))-1, max(xVal(1,:))+1, 200);
x2_range = linspace(min(xVal(2,:))-1, max(xVal(2,:))+1, 200);
[X1, X2] = meshgrid(x1_range, x2_range);

Z = zeros(size(X1));
for i = 1:numel(X1)
    x_grid = [X1(i); X2(i)];
    p_x_L0_grid = w01*evalGaussian(x_grid, m01, C01) + w02*evalGaussian(x_grid, m02, C02);
    p_x_L1_grid = w11*evalGaussian(x_grid, m11, C11) + w12*evalGaussian(x_grid, m12, C12);
    Z(i) = log(p_x_L1_grid / p_x_L0_grid);
end

figure(4), clf;
contour(X1, X2, Z, [log(gamma_theoretical), log(gamma_theoretical)], 'k-', 'LineWidth', 2.5); hold on;
plot(xVal(1, labelsVal==0), xVal(2, labelsVal==0), 'ob', 'MarkerSize', 3);
plot(xVal(1, labelsVal==1), xVal(2, labelsVal==1), '+r', 'MarkerSize', 3);
axis equal, grid on;
xlabel('x_1'), ylabel('x_2');
title('Part 1: Decision Boundary (Theoretical Optimal)');
legend('Decision Boundary', 'Class 0', 'Class 1');

% PART 2A: LOGISTIC-LINEAR MODELS

fprintf('=== PART 2A: LOGISTIC-LINEAR MODELS ===\n');
fprintf('Model: h(x,w) = 1/(1+exp(-w''*[1; x]))\n\n');

epsilon = 1e-3;
alpha = 1e-2;

datasets = {xTrain50, xTrain500, xTrain5000};
labels_datasets = {labelsTrain50, labelsTrain500, labelsTrain5000};
N_train = [50, 500, 5000];

Perror_linear = zeros(1, 3);
w_linear = cell(1, 3);

for idx = 1:3
    fprintf('Training on N=%d samples...\n', N_train(idx));
    
    xTrain = datasets{idx};
    labelsTrain = labels_datasets{idx};
    
    % Train - Pattern adapted from sample code (gradientDescent_binaryCrossEntropy.m) and discussion with Claude
    [w_linear{idx}] = trainLogisticLinear(xTrain, labelsTrain, alpha, epsilon);
    
    fprintf('  Trained weights: [%.4f, %.4f, %.4f]\n', w_linear{idx});
    
    % Evaluate - Pattern adapted from sample code (logisticGeneralizedLinearModel.m) and discussion with Claude
    h_val = logisticLinear(xVal, w_linear{idx});
    decisions_val = (h_val >= 0.5);
    
    Perror_linear(idx) = sum(decisions_val ~= labelsVal) / N;
    fprintf('  P(error) on validation: %.4f (%.2f%%)\n\n', Perror_linear(idx), 100*Perror_linear(idx));
end

% Visualize logistic-linear results
figure(5), clf;
for idx = 1:3
    xTrain = datasets{idx};
    labelsTrain = labels_datasets{idx};
    
    subplot(2, 3, idx);
    plot(xTrain(1,labelsTrain==0), xTrain(2,labelsTrain==0), 'ob', 'MarkerSize', 4); hold on;
    plot(xTrain(1,labelsTrain==1), xTrain(2,labelsTrain==1), '+r', 'MarkerSize', 4);
    
    w = w_linear{idx};
    x1_line = [min(xTrain(1,:))-1, max(xTrain(1,:))+1];
    x2_line = -(w(1) + w(2)*x1_line) / w(3);
    plot(x1_line, x2_line, 'k-', 'LineWidth', 2);
    
    axis equal, grid on;
    xlabel('x_1'), ylabel('x_2');
    title(sprintf('Linear Training N=%d', N_train(idx)));
    
    subplot(2, 3, idx+3);
    h_val = logisticLinear(xVal, w_linear{idx});
    decisions_val = (h_val >= 0.5);
    correct = (decisions_val == labelsVal);
    
    plot(xVal(1,correct & labelsVal==0), xVal(2,correct & labelsVal==0), 'og', 'MarkerSize', 2); hold on;
    plot(xVal(1,~correct & labelsVal==0), xVal(2,~correct & labelsVal==0), 'or', 'MarkerSize', 4);
    plot(xVal(1,correct & labelsVal==1), xVal(2,correct & labelsVal==1), '+g', 'MarkerSize', 2);
    plot(xVal(1,~correct & labelsVal==1), xVal(2,~correct & labelsVal==1), '+r', 'MarkerSize', 4);
    plot(x1_line, x2_line, 'k-', 'LineWidth', 2);
    
    axis equal, grid on;
    xlabel('x_1'), ylabel('x_2');
    title(sprintf('Linear Validation P(error)=%.4f', Perror_linear(idx)));
end
sgtitle('Part 2A: Logistic-Linear Model Results');

% PART 2B: LOGISTIC-QUADRATIC MODELS

fprintf('PART 2B: LOGISTIC-QUADRATIC MODELS\n');
fprintf('Model: h(x,w) = 1/(1+exp(-w''*[1; x1; x2; x1^2; x1*x2; x2^2]))\n\n');

Perror_quadratic = zeros(1, 3);
w_quadratic = cell(1, 3);

for idx = 1:3
    fprintf('Training on N=%d samples\n', N_train(idx));
    
    xTrain = datasets{idx};
    labelsTrain = labels_datasets{idx};
    
    % Train - Pattern adapted from sample code (gradientDescent_binaryCrossEntropy.m) and discussion with Claude
    [w_quadratic{idx}] = trainLogisticQuadratic(xTrain, labelsTrain, alpha*0.5, epsilon);
    
    fprintf('  Trained weights: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', w_quadratic{idx});
    
    % Evaluate - Pattern adapted from sample code (logisticGeneralizedLinearModel.m) and discussion with Claude
    h_val = logisticQuadratic(xVal, w_quadratic{idx});
    decisions_val = (h_val >= 0.5);
    
    Perror_quadratic(idx) = sum(decisions_val ~= labelsVal) / N;
    fprintf('  P(error) on validation: %.4f (%.2f%%)\n\n', Perror_quadratic(idx), 100*Perror_quadratic(idx));
end

% Visualize logistic-quadratic results
figure(6), clf;
for idx = 1:3
    xTrain = datasets{idx};
    labelsTrain = labels_datasets{idx};
    
    subplot(2, 3, idx);
    plot(xTrain(1,labelsTrain==0), xTrain(2,labelsTrain==0), 'ob', 'MarkerSize', 4); hold on;
    plot(xTrain(1,labelsTrain==1), xTrain(2,labelsTrain==1), '+r', 'MarkerSize', 4);
    
    x1_grid = linspace(min(xTrain(1,:))-1, max(xTrain(1,:))+1, 100);
    x2_grid = linspace(min(xTrain(2,:))-1, max(xTrain(2,:))+1, 100);
    [X1_grid, X2_grid] = meshgrid(x1_grid, x2_grid);
    
    H_grid = zeros(size(X1_grid));
    for i = 1:numel(X1_grid)
        x_test = [X1_grid(i); X2_grid(i)];
        H_grid(i) = logisticQuadratic(x_test, w_quadratic{idx});
    end
    
    contour(X1_grid, X2_grid, H_grid, [0.5, 0.5], 'k-', 'LineWidth', 2);
    axis equal, grid on;
    xlabel('x_1'), ylabel('x_2');
    title(sprintf('Quadratic Training N=%d', N_train(idx)));
    
    subplot(2, 3, idx+3);
    h_val = logisticQuadratic(xVal, w_quadratic{idx});
    decisions_val = (h_val >= 0.5);
    correct = (decisions_val == labelsVal);
    
    plot(xVal(1,correct & labelsVal==0), xVal(2,correct & labelsVal==0), 'og', 'MarkerSize', 2); hold on;
    plot(xVal(1,~correct & labelsVal==0), xVal(2,~correct & labelsVal==0), 'or', 'MarkerSize', 4);
    plot(xVal(1,correct & labelsVal==1), xVal(2,correct & labelsVal==1), '+g', 'MarkerSize', 2);
    plot(xVal(1,~correct & labelsVal==1), xVal(2,~correct & labelsVal==1), '+r', 'MarkerSize', 4);
    
    x1_grid_val = linspace(min(xVal(1,:))-0.5, max(xVal(1,:))+0.5, 100);
    x2_grid_val = linspace(min(xVal(2,:))-0.5, max(xVal(2,:))+0.5, 100);
    [X1_val, X2_val] = meshgrid(x1_grid_val, x2_grid_val);
    
    H_grid_val = zeros(size(X1_val));
    for i = 1:numel(X1_val)
        x_test = [X1_val(i); X2_val(i)];
        H_grid_val(i) = logisticQuadratic(x_test, w_quadratic{idx});
    end
    
    contour(X1_val, X2_val, H_grid_val, [0.5, 0.5], 'k-', 'LineWidth', 2);
    axis equal, grid on;
    xlabel('x_1'), ylabel('x_2');
    title(sprintf('Quadratic Validation P(error)=%.4f', Perror_quadratic(idx)));
end
sgtitle('Part 2B: Logistic-Quadratic Model Results');

% COMPARISON AND DISCUSSION

fprintf('\n FINAL COMPARISON\n');
fprintf('Classifier Performance on Validation Set (10K samples):\n\n');
fprintf('                     N=50      N=500     N=5000\n');
fprintf('Theoretical Optimal: %.4f    %.4f    %.4f\n', min_Perror, min_Perror, min_Perror);
fprintf('Logistic-Linear:     %.4f    %.4f    %.4f\n', Perror_linear);
fprintf('Logistic-Quadratic:  %.4f    %.4f    %.4f\n\n', Perror_quadratic);

% Summary plot
figure(7), clf;
semilogx([50, 500, 5000], [min_Perror, min_Perror, min_Perror], 'k-', 'LineWidth', 2); hold on;
semilogx([50, 500, 5000], Perror_linear, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
semilogx([50, 500, 5000], Perror_quadratic, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Training Set Size (N)');
ylabel('P(error) on Validation Set');
title('Performance Comparison: All Classifiers');
legend('Theoretical Optimal (Part 1) - Independent of N', ...
       'Logistic-Linear (Part 2A)', ...
       'Logistic-Quadratic (Part 2B)', ...
       'Location', 'best');
grid on;

fprintf('KEY OBSERVATIONS:\n');
fprintf('1. Theoretical optimal achieves min P(error) = %.4f\n', min_Perror);
fprintf('2. Linear model performance:\n');
fprintf('   - Underfits GMM structure (linear boundary for nonlinear data)\n');
fprintf('   - Best at N=5000: P(error) = %.4f (gap = %.4f)\n', Perror_linear(3), Perror_linear(3)-min_Perror);
fprintf('3. Quadratic model performance:\n');
fprintf('   - Better captures GMM structure\n');
fprintf('   - Best at N=5000: P(error) = %.4f (gap = %.4f)\n', Perror_quadratic(3), Perror_quadratic(3)-min_Perror);
fprintf('4. Both models improve with more training data (bias-variance tradeoff)\n');
fprintf('5. Quadratic model shows overfitting at N=50 but improves with more data\n\n');


% HELPER FUNCTIONS

% Code for evalGaussian function taken from sample code (evalGaussian.m - used verbatim)
function g = evalGaussian(x,mu,Sigma)
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

% Code for generateDataFromGMM function taken from sample code (generateDataFromGMM.m - used verbatim)
function [x,labels] = generateDataFromGMM(N,gmmParameters)
priors = gmmParameters.priors;
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1);
C = length(priors);
x = zeros(n,N); labels = zeros(1,N); 
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl);
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end

% Code for ROCcurve function from my ASSIGN1_Q1.m (originally adapted from sample code LDAwithROCcurve.m)
function [Pfp,Ptp,Pfn,Perror,thresholdList] = ROCcurve(discriminantScores,label, p)
[sortedScores,~] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Pfp(i) = length(find(decisions==1 & label==0))/length(find(label==0));
    Ptp(i) = length(find(decisions==1 & label==1))/length(find(label==1));
    Pfn(i) = 1 - Ptp(i);
    Perror(i) = p(1)*Pfp(i) + p(2)*Pfn(i);
end
end

% Logistic-Linear model evaluation - Pattern adapted from sample code (logisticGeneralizedLinearModel.m) and discussion with Claude
function h = logisticLinear(x, w)
N = size(x, 2);
z = [ones(1, N); x];
h = 1 ./ (1 + exp(-w' * z));
end

% Logistic-Quadratic model evaluation - Pattern adapted from sample code (logisticGeneralizedLinearModel.m) and discussion with Claude
function h = logisticQuadratic(x, w)
N = size(x, 2);
z = [ones(1, N); x(1,:); x(2,:); x(1,:).^2; x(1,:).*x(2,:); x(2,:).^2];
h = 1 ./ (1 + exp(-w' * z));
end

% Training function for logistic-linear model - Pattern adapted from sample code (gradientDescent_binaryCrossEntropy.m) and discussion with Claude
function w = trainLogisticLinear(x, labels, alpha, epsilon)
N = size(x, 2);
w = randn(3, 1);
minIter = 100;
iterCount = 0;

while true
    z = [ones(1, N); x];
    h = 1 ./ (1 + exp(-w' * z));
  
    % Gradient computation - Pattern from sample code (binaryCrossEntropyCostFunction.m)
    gradient = z * (h - labels)' / N;
    
    w = w - alpha * gradient;
    iterCount = iterCount + 1;
    
    if iterCount >= minIter && norm(gradient) < epsilon
        break;
    end
    
    if iterCount > 10000
        warning('Maximum iterations reached without convergence');
        break;
    end
end
end

% Training function for logistic-quadratic model - Pattern adapted from sample code (gradientDescent_binaryCrossEntropy.m) and discussion with Claude
function w = trainLogisticQuadratic(x, labels, alpha, epsilon)
N = size(x, 2);
w = randn(6, 1);
minIter = 100;
iterCount = 0;

while true
    z = [ones(1, N); x(1,:); x(2,:); x(1,:).^2; x(1,:).*x(2,:); x(2,:).^2];
    h = 1 ./ (1 + exp(-w' * z));
    
    % Gradient computation - Pattern from sample code (binaryCrossEntropyCostFunction.m)
    gradient = z * (h - labels)' / N;
    
    w = w - alpha * gradient;
    iterCount = iterCount + 1;
    
    if iterCount >= minIter && norm(gradient) < epsilon
        break;
    end
    
    if iterCount > 10000
        warning('Maximum iterations reached without convergence');
        break;
    end
end
end