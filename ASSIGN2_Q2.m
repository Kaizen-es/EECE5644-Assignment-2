%{
                    EECE5644 FALL 2025 - ASSIGNMENT 2
                                QUESTION 2
              POLYNOMIAL REGRESSION: ML vs MAP ESTIMATION

NOTE ON COLLABORATION:
This implementation builds upon Professor's provided code:
- Professor's hw2q2.m data generation function (used verbatim)
- Professor's generateDataFromGMM.m (used verbatim)
- Professor's PolynomialFitCrossValidation.m formPsiX function (pattern adapted)
- Professor's PolynomialFitCrossValidation.m visualization structure (pattern adapted)

Theoretical foundation from Murphy (2012):
- Section 7.3 (Linear Regression) for ML estimator
- Section 7.5 (Ridge Regression) for MAP estimator

Implementation developed through collaboration with ChatGPT and Claude AI:
- Design matrix construction for cubic polynomial
- Noise variance estimation strategy
- Hyperparameter grid search approach
- Coefficient analysis output formatting
- Analysis and discussion

All implementation decisions and final analysis represent my own work
synthesized from these resources and course materials.
%}

clear all, close all,

fprintf('\nASSIGNMENT 2 - QUESTION 2: POLYNOMIAL REGRESSION\n');

% GENERATE DATA

Ntrain = 100;
Nvalidate = 1000;

% Data generation using Professor's hw2q2.m function (used verbatim, see bottom)
[xTrain, yTrain, xValidate, yValidate] = hw2q2(Ntrain, Nvalidate);

fprintf('Dataset generated:\n');
fprintf('  Training: %d samples\n', Ntrain);
fprintf('  Validation: %d samples\n\n', Nvalidate);

% BUILD DESIGN MATRIX

% Cubic polynomial features
% Design matrix construction adapted from Professor's PolynomialFitCrossValidation.m formPsiX function
% Extended to cubic terms with ChatGPT assistance
PsiTrain = buildDesignMatrix(xTrain);
PsiValidate = buildDesignMatrix(xValidate);

[Ntrain, p] = size(PsiTrain);
fprintf('Design matrix: %d samples x %d features (cubic polynomial)\n\n', Ntrain, p);

% ML ESTIMATOR (Ordinary Least Squares)

fprintf('ML ESTIMATOR (Ordinary Least Squares)\n');

% ML solution: w = (Psi'*Psi)^(-1)*Psi'*y
% Murphy (2012) Eq. 7.11
% Implementation pattern from Professor's PolynomialFitCrossValidation.m fitPolynomial function
w_ML = (PsiTrain' * PsiTrain) \ (PsiTrain' * yTrain');

% Training error
% MSE calculation pattern from Professor's PolynomialFitCrossValidation.m calculateMSE function
yPred_train_ML = PsiTrain * w_ML;
MSE_train_ML = mean((yTrain' - yPred_train_ML).^2);

% Validation error
yPred_val_ML = PsiValidate * w_ML;
MSE_val_ML = mean((yValidate' - yPred_val_ML).^2);

fprintf('  Training MSE: %.4f\n', MSE_train_ML);
fprintf('  Validation MSE: %.4f\n', MSE_val_ML);

% Estimate noise variance
% Unbiased estimator strategy suggested by ChatGPT
% sigma^2 = ||y - Psi*w_ML||^2 / (N - p)
residuals_ML = yTrain' - yPred_train_ML;
sigma2_hat = sum(residuals_ML.^2) / (Ntrain - p);
fprintf('  Estimated σ²: %.4f\n\n', sigma2_hat);

% MAP ESTIMATOR (Ridge Regression)

fprintf('MAP ESTIMATOR (Ridge Regression)\n');
fprintf('Prior: w ~ N(0, γI)\n');
fprintf('Solution: w_MAP = (Psi''*Psi + λI)^(-1)*Psi''*y where λ = σ²/γ\n\n');

% Hyperparameter search range suggested by ChatGPT
gamma_values = 10.^linspace(-12, 3, 101);
MSE_train_MAP = zeros(size(gamma_values));
MSE_val_MAP = zeros(size(gamma_values));

fprintf('Evaluating MAP for %d values of γ ∈ [10^-6, 10^6]...\n', length(gamma_values));

% Hyperparameter loop structure adapted from Professor's PolynomialFitCrossValidation.m
% (lines trying all polynomial orders M = 1:maxM)
for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    lambda = sigma2_hat / gamma;
    
    % MAP solution - Murphy (2012) Eq. 7.87
    w_MAP = (PsiTrain' * PsiTrain + lambda * eye(p)) \ (PsiTrain' * yTrain');
    
    % Training MSE - calculation from Professor's PolynomialFitCrossValidation.m calculateMSE function
    yPred_train = PsiTrain * w_MAP;
    MSE_train_MAP(i) = mean((yTrain' - yPred_train).^2);
    
    % Validation MSE
    yPred_val = PsiValidate * w_MAP;
    MSE_val_MAP(i) = mean((yValidate' - yPred_val).^2);
end

% Find optimal gamma - selection pattern from Professor's PolynomialFitCrossValidation.m
% [~,bestM] = min(AverageMSEvalidate)
[min_MSE_val, idx_opt] = min(MSE_val_MAP);
gamma_opt = gamma_values(idx_opt);
lambda_opt = sigma2_hat / gamma_opt;

fprintf('  Optimal γ* = %.4e (log10(γ*) = %.2f)\n', gamma_opt, log10(gamma_opt));
fprintf('  Corresponding λ* = %.4e\n', lambda_opt);
fprintf('  Validation MSE at γ*: %.4f\n\n', min_MSE_val);

% Train final MAP model with optimal gamma
w_MAP_opt = (PsiTrain' * PsiTrain + lambda_opt * eye(p)) \ (PsiTrain' * yTrain');
yPred_train_MAP = PsiTrain * w_MAP_opt;
MSE_train_MAP_opt = mean((yTrain' - yPred_train_MAP).^2);

fprintf('  Training MSE at γ*: %.4f\n\n', MSE_train_MAP_opt);

% COEFFICIENT ANALYSIS
% Output formatting developed with ChatGPT
% Shows regularization effect on learned parameters

fprintf('=== COEFFICIENT ANALYSIS ===\n');
fprintf('Features: [1, x1, x2, x1², x1x2, x2², x1³, x1²x2, x1x2², x2³]\n\n');

fprintf('ML Coefficients:\n');
for i = 1:p
    fprintf('  w_%d = %10.4f\n', i-1, w_ML(i));
end
fprintf('  ||w_ML||₂ = %.4f\n\n', norm(w_ML));

fprintf('MAP Coefficients (γ* = %.2e, λ* = %.2e):\n', gamma_opt, lambda_opt);
for i = 1:p
    fprintf('  w_%d = %10.4f\n', i-1, w_MAP_opt(i));
end
fprintf('  ||w_MAP||₂ = %.4f\n', norm(w_MAP_opt));
fprintf('  Coefficient norm reduction: %.1f%%\n\n', ...
    100*(1 - norm(w_MAP_opt)/norm(w_ML)));

% COMPARISON

fprintf('COMPARISON: ML vs MAP\n');
fprintf('                Training MSE    Validation MSE\n');
fprintf('ML (OLS):       %.4f          %.4f\n', MSE_train_ML, MSE_val_ML);
fprintf('MAP (Ridge):    %.4f          %.4f\n', MSE_train_MAP_opt, min_MSE_val);
fprintf('Improvement:    %.4f          %.4f\n\n', ...
    MSE_train_ML - MSE_train_MAP_opt, MSE_val_ML - min_MSE_val);

fprintf('ANALYSIS\n');
fprintf('Coefficient norms:\n');
fprintf('  ||w_ML|| = %.4f\n', norm(w_ML));
fprintf('  ||w_MAP|| = %.4f\n', norm(w_MAP_opt));
fprintf('  Shrinkage: %.2f%%\n\n', 100*(1 - norm(w_MAP_opt)/norm(w_ML)));

fprintf('Relationship between ML and MAP:\n');
fprintf('  As γ → ∞: λ → 0, MAP → ML\n');
fprintf('  As γ → 0: λ → ∞, coefficients → 0\n');
fprintf('  Optimal γ* balances bias-variance tradeoff\n\n');

% VISUALIZATION

% Plot MSE vs gamma
% Visualization structure adapted from Professor's PolynomialFitCrossValidation.m
% (semilogy plot of AverageMSEtrain and AverageMSEvalidate vs model order)
figure(3), clf;
loglog(gamma_values, MSE_train_MAP, 'b-', 'LineWidth', 2); hold on;
loglog(gamma_values, MSE_val_MAP, 'r-', 'LineWidth', 2); hold on;
loglog(gamma_opt, min_MSE_val, 'go', 'MarkerSize', 10, 'LineWidth', 2);

% Add ML performance as horizontal line for reference
loglog([min(gamma_values), max(gamma_values)], [MSE_val_ML, MSE_val_ML], ...
    'k--', 'LineWidth', 1.5);

xlabel('Prior Variance γ');
ylabel('Mean Squared Error');
title('MAP Performance vs Hyperparameter γ');
legend('MAP Training MSE', 'MAP Validation MSE', ...
    sprintf('Optimal γ* = %.2e', gamma_opt), 'ML Validation MSE', ...
    'Location', 'best');

grid on;

% Plot MSE vs lambda
lambda_values = sigma2_hat ./ gamma_values;

figure(4), clf;
semilogx(lambda_values, MSE_train_MAP, 'b-', 'LineWidth', 2); hold on;
semilogx(lambda_values, MSE_val_MAP, 'r-', 'LineWidth', 2);
semilogx(lambda_opt, min_MSE_val, 'go', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Regularization Parameter λ = σ²/γ');
ylabel('Mean Squared Error');
title('Ridge Regression: MSE vs λ');
legend('Training MSE', 'Validation MSE', sprintf('Optimal λ* = %.2e', lambda_opt), ...
    'Location', 'best');
grid on;

% Scatter plot: Actual vs Predicted for ML
figure(5), clf;
subplot(1,2,1);
plot(yValidate, yPred_val_ML, 'b.', 'MarkerSize', 8);
hold on;
minY = min([yValidate, yPred_val_ML']);
maxY = max([yValidate, yPred_val_ML']);
plot([minY, maxY], [minY, maxY], 'k--', 'LineWidth', 1.5);
xlabel('True y');
ylabel('Estimated y');
title(sprintf('ML Estimator\nValidation MSE = %.4f', MSE_val_ML));
axis equal;
grid on;

% Scatter plot: Actual vs Predicted for MAP
yPred_val_MAP_opt = PsiValidate * w_MAP_opt;
subplot(1,2,2);
plot(yValidate, yPred_val_MAP_opt, 'r.', 'MarkerSize', 8);
hold on;
minY = min([yValidate, yPred_val_MAP_opt']);
maxY = max([yValidate, yPred_val_MAP_opt']);
plot([minY, maxY], [minY, maxY], 'k--', 'LineWidth', 1.5);
xlabel('True y');
ylabel('Estimated y');
title(sprintf('MAP Estimator (γ* = %.2e)\nValidation MSE = %.4f', gamma_opt, min_MSE_val));
axis equal;
grid on;

sgtitle('Actual vs Estimated Values on Validation Set');


% HELPER FUNCTIONS

% Design matrix construction for cubic polynomial
% Function structure adapted from Professor's PolynomialFitCrossValidation.m formPsiX function
% Extended to cubic terms with ChatGPT assistance
function Psi = buildDesignMatrix(x)
% Builds design matrix for cubic polynomial
% Input:  x - 2 x N matrix
% Output: Psi - N x 10 design matrix
% Features: [1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3]

N = size(x, 2);
x1 = x(1, :)';
x2 = x(2, :)';

% Feature construction pattern from Professor's PolynomialFitCrossValidation.m formPsiX:
% PsiX(1,:) = ones(1,N); for m = 1:M, PsiX(m+1,:) = x.^m; end
% Extended to 2D cubic polynomial
Psi = [ones(N, 1), x1, x2, x1.^2, x1.*x2, x2.^2, ...
       x1.^3, (x1.^2).*x2, x1.*(x2.^2), x2.^3];
end

% Code for hw2q2 function taken from Professor's hw2q2.m (used verbatim)
function [xTrain,yTrain,xValidate,yValidate] = hw2q2(Ntrain,Nvalidate)
data = generateData(Ntrain);
figure(1), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset'),
xTrain = data(1:2,:); yTrain = data(3,:);

data = generateData(Nvalidate);
figure(2), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Validation Dataset'),
xValidate = data(1:2,:); yValidate = data(3,:);
end

% Code for generateData function taken from Professor's hw2q2.m (used verbatim)
function x = generateData(N)
gmmParameters.priors = [.3,.4,.3];
gmmParameters.meanVectors = [-10 0 10;0 0 0;10 0 -10];
gmmParameters.covMatrices(:,:,1) = [1 0 -3;0 1 0;-3 0 15];
gmmParameters.covMatrices(:,:,2) = [8 0 0;0 .5 0;0 0 .5];
gmmParameters.covMatrices(:,:,3) = [1 0 -3;0 1 0;-3 0 15];
[x,labels] = generateDataFromGMM(N,gmmParameters);
end

% Code for generateDataFromGMM function taken from Professor's generateDataFromGMM.m (used verbatim)
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