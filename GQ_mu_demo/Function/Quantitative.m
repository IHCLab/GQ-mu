function [Phi_en, RMSE_score, Phi_ab, B_estimated, S_estimated] = ...
    Quantitative(B_gt, S_gt, B_estimated, S_estimated, use_signatue)

N = size(B_gt, 2);

if use_signatue == 1
    
    % Signature-based evaluation
    [Phi_en, B_estimated, perm_mtx] = SAE_FAAE(B_gt, B_estimated);
    
    S_estimated = S_estimated' * perm_mtx;
    S_estimated = S_estimated';
    
    RMSE_score = rmse(S_gt, S_estimated, "all");

else
    
    % Abundance-based evaluation
    [RMSE_score, S_estimated, perm_mtx] = SAE_RMSE(S_gt', S_estimated');
    S_estimated = S_estimated';
    
    [Phi_ab, ~, ~] = SAE_FAAE(S_gt', S_estimated');
    
    B_estimated = B_estimated * perm_mtx;
    
    Phi_en = norm( ...
        acos( ...
            diag(B_gt' * B_estimated) ./ ...
            ((sum(B_gt.^2) .* sum(B_estimated.^2))'.^0.5) ...
        ) ...
    )^2 / N;
    
    Phi_en = Phi_en^0.5;
    Phi_en = Phi_en * 180 / pi;

end