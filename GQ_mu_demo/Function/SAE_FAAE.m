function [d_wass, Y_sub, perm_mtx] = SAE_FAAE(X, Y)

    n = size(X, 2);
    m = size(Y, 2);
    r = min(m, n);

    if n > m
        index = nchoosek(1:n, m);

        temp = Y;
        Y    = X;
        X    = temp;

    elseif m > n
        index = nchoosek(1:m, n);

    else  % m == n
        index = 1:m;
    end

    L = size(index, 1);

    d_wass = zeros(L, 1);

    for j = 1:L

        Y_sub = Y(:, index(j, :));

        CRD = corrcoef([X, Y_sub]);
        D   = abs(CRD(r+1:2*r, 1:r));

        % ================= Permutation =================
        perm_mtx = zeros(r, r);
        aux      = zeros(r, 1);

        for i = 1:r
            [ld, cd] = find(max(D(:)) == D);
            ld = ld(1);
            cd = cd(1);  % in the case of more than one maximum

            perm_mtx(ld, cd) = 1;

            D(:, cd) = aux;
            D(ld, :) = aux';
        end

        Y_sub = Y_sub * perm_mtx;

        d_wass(j) = ( ...
            norm(acos(diag(X' * Y_sub) ./ ((sum(X.^2) .* sum(Y_sub.^2))'.^0.5)))^2 / r ...
        )^0.5;

    end

    d_wass = min(d_wass);
    d_wass = d_wass * 180 / pi;

end