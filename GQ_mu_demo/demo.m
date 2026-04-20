close all; clear all; clc; rng('default');
addpath(genpath("Function/")) 
addpath(genpath("dataset/"))

%% Experimental Settings 
N = 6; % Number of Multispectral Signautres
data_name = 'Ottawa';
load(strcat(data_name,'.mat')); % Obtain High-quality HSI for simulation 
[MSI, B_gt, S_gt, D] = generating_MSI(HrHSI, N);
%% Parameter Settings for GQ-mu 

par.lambda1 = 0.001; % for|| S ||_1
par.lambda2 = 0.01; % for || S -S_Qu ||_F
par.lambda3 = 1000;% for|| (A-c1^T) sqrt(W) ||_F
par.lambda4 = 100; % for|| A-A^(t-1)||_F
par.QAT = 'Test'; % Set 'Train' for training QDIP 
par.data_name = data_name;
%% Underdetermined Multispectral Unmixing

[B_Gmu,S_Gmu,time_QGmu] =  GQ_mu(MSI,D,N,par);

%% Quantative Evaluation
 [Phi_en, RMSE_score, Phi_ab, B_estimated, S_estimated] = ...
    Quantitative(B_gt, S_gt, B_Gmu, S_Gmu, 0);

fprintf('\nphi_ab: %.3f\n', Phi_ab)
fprintf('phi_en: %.3f\n', Phi_en)
fprintf('RMSE: %.3f\n\n', RMSE_score)

%% Reshape abundance maps
[row, col,~] = size(MSI);
S_estimated_show = reshape(S_estimated', row, col, []);
S_gt_show        = reshape(S_gt', row, col, []);

%% Plot signatures and abundances in one figure
figure('Name', 'Qualitative Comparison');

map = 'gray';

for i = 1:N
    
    % -----------------------------
    % Row 1: Signature comparison
    % -----------------------------
    subplot(3, N, i)
    plot(B_estimated(:, i), '--o','LineWidth', 2.5);
    hold on;
    plot(B_gt(:, i), '--square','LineWidth', 2.5);
    hold off;
    
    title(['Signature ', num2str(i)]);
    if i == 1
        ylabel('Signature', 'FontSize', 14);
        legend('Estimated', 'GT', 'Location', 'best');
    end
    axis tight;
    grid on;
    
    % -----------------------------
    % Row 2: Estimated abundance
    % -----------------------------
    subplot(3, N, i + N)
    imshow(ImGray2Pseudocolor(S_estimated_show(:, :, i), map, 255));
    title(['Estimated ', num2str(i)]);
    if i == 1
        ylabel('Estimated', 'FontSize', 14);
    end
    
    % -----------------------------
    % Row 3: GT abundance
    % -----------------------------
    subplot(3, N, i + 2*N)
    imshow(ImGray2Pseudocolor(S_gt_show(:, :, i), map, 255));
    title(['GT ', num2str(i)]);
    if i == 1
        ylabel('GT', 'FontSize', 14);
    end

end

set(gcf, 'Position', [100, 100, 1400, 500]);