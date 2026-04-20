function [MSI, B_reference, S_reference, D] = generating_MSI(HrHSI, N)
%% Obtain reference hyperspectral signatures and abundances
HSI = crop_data(HrHSI);
[r, c, b] = size(HSI);

HSI_2D = reshape(HSI, r*c, b)';
fprintf('Starting HiSun BSS process...\n\n')
[A_reference, S_reference] = HiSun(HSI_2D, N);
fprintf('\n')
fprintf('Reference hyperspectral signatures obtained.\n')
%% Spectral downsampling (simulate MSI)
banduse_Landsat = [ ...
     1   6;
     7  15;
    19  26;
    34  68
];

MSI_band = size(banduse_Landsat, 1);
SRF = estD(banduse_Landsat, b);

B_reference = SRF * A_reference;
MSI_2D = B_reference * S_reference;

% Add slight Gaussian Deviation
MSI_2D = MSI_2D + 1e-4 * randn(size(MSI_2D));

MSI = reshape(MSI_2D', r, c, []);

%% SRF for unsupervised spectral augmentation
banduse = obtain_band(MSI_band, 2);
D = estD(banduse, MSI_band * 2) * 2;

end


%% Sub-function 1
function X = crop_data(X)
% Normalize and crop
nor = @(R) (R - min(R(:))) / (max(R(:)) - min(R(:)));
X = nor(X);
X = X(1:256, 1:256, :);
end


%% Sub-function 2
function D = estD(banduse, bands_ref)
% Construct spectral response function (uniform averaging)
bands_m = size(banduse, 1);
D = zeros(bands_m, bands_ref);

for b = 1:bands_m
    D(b, banduse(b,1):banduse(b,2)) = ...
        1 / (banduse(b,2) - banduse(b,1) + 1);
end

end


%% Sub-function 3
function banduse_2 = obtain_band(MSI_band, scale)
% Generate uniform band intervals
banduse_2 = zeros(MSI_band, 2);
banduse_2(:,1) = 1:scale:scale*MSI_band;
banduse_2(:,2) = banduse_2(:,1) + scale - 1;
end