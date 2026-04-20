function [d_wass, Y_sub, perm_mtx] = SAE_RMSE(X, Y)

    n = size(X, 2);
    m = size(Y, 2);
    r = min(m, n);

    if n > m
        index = nchoosek(1:n, m);
        temp  = Y;
        Y     = X;
        X     = temp;

    elseif m > n
        index = nchoosek(1:m, n);

    else % m == n
        index = 1:m;
    end

    L = size(index, 1);
    d_wass = zeros(L, 1);

    for j = 1:L

        Y_sub = Y(:, index(j, :));

        CRD = corrcoef([X Y_sub]);
        D   = abs(CRD(r+1:2*r, 1:r));

        % ================= Permutation =================
        perm_mtx = zeros(r, r);
        aux      = zeros(r, 1);

        for i = 1:r
            [ld, cd] = find(max(D(:)) == D);
            ld = ld(1);
            cd = cd(1); % in case of multiple maxima

            perm_mtx(ld, cd) = 1;

            D(:, cd) = aux;
            D(ld, :) = aux';
        end

        Y_sub = Y_sub * perm_mtx;

    end

    d_wass = RMSE_MINE(X, Y_sub);

end


%% ================= RMSE =================
function rmse_mine = RMSE_MINE(X, Y_sub)

    [pixel, N] = size(X);

    rmse_mine = sqrt( ...
        sum((X - Y_sub).^2, 'all') / (pixel * N) ...
    ) * 100;

end