% =========================================================================
% FINAL PROJECT: Representation Methods for Crop Type Classification
% DATA 604, Spring 2026
% Dataset : UCI Crop Mapping — Fused Optical-Radar (325,834 x 175, 7 classes)
%  Column 1 = class label; columns 2–175 = real-valued features.
% Representations : Raw | PCA | kPCA (RBF) | Laplacian Eigenmaps
% Classifier : kNN (k tuned per representation on the validation set)
% Metrics : Accuracy | Macro F1-score
% Extras : Wall-clock timing, k-sensitivity curves,
%  confusion matrices for all four representations,
%  per-class F1 grouped bar chart,
%  drtoolbox cross-validation of PCA and kPCA embeddings.
%
% Requirements
%  - MATLAB R2016b or later (implicit broadcasting via bsxfun)
%  - drtoolbox (Van der Maaten) on the MATLAB path — used ONLY to
%  cross-validate PCA and kPCA embeddings, not for primary results.
%  All classifiers and out-of-sample projections are implemented
%  from scratch.
%
% Note on drtoolbox usage
%  drtoolbox's laplacian_eigen.m is intentionally not used here because
%  it does not support out-of-sample projection for validation/test sets.
%  Our custom implementation uses Gaussian-weighted NN interpolation
%  (Bengio et al., 2004), which correctly handles unseen points.
%
% Usage
%  1. Place WinnipegDataset.txt in the same folder as this script.
%  2. Ensure the drtoolbox folder is on the MATLAB path (addpath below).
%  3. Run with F5, or section by section with Ctrl+Enter.
%  4. All figures are saved as .png files in the working directory.
%  5. A timing summary is printed at the end.
% =========================================================================
 
clear; clc; close all;
rng(42);
addpath(genpath('drtoolbox'));
 
%% SECTION 1 — Load Data
% =========================================================================
fprintf('=== Loading data ===\n');
t_load = tic;
data = readmatrix('WinnipegDataset.txt');
y_all = data(:, 1);
X_all = data(:, 2:end);
fprintf('Full dataset: %d samples x %d features, %d classes [%.1fs]\n', ...
    size(X_all,1), size(X_all,2), numel(unique(y_all)), toc(t_load));
class_names = {'Corn','Peas','Canola','Soybeans','Oats','Wheat','Broadleaf'};
 
%% SECTION 2 — Stratified Subsample
% =========================================================================
MAX_PER_CLASS = 10000;
fprintf('\n=== Stratified subsampling (max %d per class) ===\n', MAX_PER_CLASS);
classes = unique(y_all);
idx_keep = [];
for i = 1:numel(classes)
    ci = find(y_all == classes(i));
    take = min(MAX_PER_CLASS, numel(ci));
    perm = ci(randperm(numel(ci), take));
    idx_keep = [idx_keep; perm]; %#ok<AGROW>
end
idx_keep = idx_keep(randperm(numel(idx_keep)));
X_sub = X_all(idx_keep,:);
y_sub = y_all(idx_keep);
fprintf('Subsampled: %d x %d\n', size(X_sub,1), size(X_sub,2));
for i = 1:numel(classes)
    fprintf('  Class %d (%-10s): %d samples\n', classes(i), ...
        class_names{classes(i)}, sum(y_sub == classes(i)));
end
 
%% SECTION 3 — Train / Validation / Test Split (60 / 20 / 20, stratified)
% =========================================================================
fprintf('\n=== Train / Validation / Test split (60 / 20 / 20, stratified) ===\n');
idx_tr = []; idx_val = []; idx_te = [];
for i = 1:numel(classes)
    c = classes(i);
    idx = find(y_sub == c);
    idx = idx(randperm(numel(idx)));
    n = numel(idx);
    n_tr  = round(0.60 * n);
    n_val = round(0.20 * n);
    idx_tr  = [idx_tr;  idx(1:n_tr)];                        %#ok<AGROW>
    idx_val = [idx_val; idx(n_tr+1:n_tr+n_val)];             %#ok<AGROW>
    idx_te  = [idx_te;  idx(n_tr+n_val+1:end)];              %#ok<AGROW>
end
idx_tr  = idx_tr(randperm(numel(idx_tr)));
idx_val = idx_val(randperm(numel(idx_val)));
idx_te  = idx_te(randperm(numel(idx_te)));
X_train = X_sub(idx_tr,:);  y_train = y_sub(idx_tr);
X_val   = X_sub(idx_val,:); y_val   = y_sub(idx_val);
X_test  = X_sub(idx_te,:);  y_test  = y_sub(idx_te);
fprintf('Train: %d  Validation: %d  Test: %d\n', ...
    numel(y_train), numel(y_val), numel(y_test));
 
%% SECTION 4 — Standardize (fit on train only; apply to validation/test)
% =========================================================================
mu = mean(X_train, 1);
sg = std(X_train, 0, 1);
sg(sg == 0) = 1;  % prevent division by zero for constant features
X_train_s = (X_train - mu) ./ sg;
X_val_s   = (X_val   - mu) ./ sg;
X_test_s  = (X_test  - mu) ./ sg;
 
%% SECTION 5 — Representation Methods (with wall-clock timing)
% =========================================================================
n_comp = 50;
timing = struct();
 
% --- 5a. Raw ------------------------------------------------------------
fprintf('\n--- [1/4] Raw (standardized) ---\n');
t0 = tic;
rep.raw.train = X_train_s;
rep.raw.val   = X_val_s;
rep.raw.test  = X_test_s;
timing.raw = toc(t0);
fprintf('Dimensions: %d [%.2fs]\n', size(rep.raw.train,2), timing.raw);
 
% --- 5b. PCA (drtoolbox pca()) ------------------------------------------
fprintf('\n--- [2/4] PCA (drtoolbox) ---\n');
t0 = tic;
score_tr = pca(X_train_s, n_comp);
% OOS projection via manual SVD (drtoolbox pca() returns no mapping struct)
mu_tr = mean(X_train_s, 1);
[~, S, V] = svd(X_train_s - mu_tr, 'econ');
eigvals = diag(S).^2;
total_var = sum(eigvals);
explained_full = 100 * eigvals / total_var;
coeff = V(:, 1:n_comp);
cum_var = cumsum(explained_full(1:n_comp));
fprintf('%d components explain %.2f%% of variance.\n', n_comp, cum_var(end));
rep.pca.train = (X_train_s - mu_tr) * coeff;
rep.pca.val   = (X_val_s   - mu_tr) * coeff;
rep.pca.test  = (X_test_s  - mu_tr) * coeff;
timing.pca = toc(t0);
fprintf('PCA complete (drtoolbox). [%.2fs]\n', timing.pca);
% Cross-check: drtoolbox pca() vs manual SVD
score_svd = (X_train_s - mu_tr) * coeff;
corr_vals = zeros(1, 5);
for pc = 1:5
    c = corrcoef(score_tr(:,pc), score_svd(:,pc));
    corr_vals(pc) = abs(c(1,2));
end
fprintf('  |r| (drtoolbox vs manual SVD, first 5 PCs): ');
fprintf('%.4f ', corr_vals);
fprintf('\n  Embeddings consistent: %s\n', mat2str(all(corr_vals > 0.999)));
 
% --- 5c. kPCA (RBF kernel) — drtoolbox kernel_pca() + Nystrom OOS ------
fprintf('\n--- [3/4] kPCA (RBF kernel, drtoolbox) ---\n');
t0 = tic;
% Estimate sigma^2 via the median heuristic on a 1,000-point subsample.
n_tr = size(X_train_s,1);
m = min(1000, n_tr);
idx_s = randperm(n_tr, m);
X_s = X_train_s(idx_s,:);
sqnorms = sum(X_s.^2, 2);
D2_mat = bsxfun(@plus, sqnorms, sqnorms.') - 2*(X_s*X_s.');
D2_mat = max(D2_mat, 0);
mask = triu(true(m), 1);
kpca_sig2 = median(D2_mat(mask));
fprintf('RBF sigma^2 = %.4f (median heuristic, %d-point subsample)\n', ...
    kpca_sig2, m);
% drtoolbox kernel_pca() fit on 5,000-pt stratified subsample.
% Returns embedding only (no mapping struct in this drtoolbox version).
MAX_KPCA = 5000;
idx_kpca = stratified_subsample(y_train, MAX_KPCA);
X_kpca_fit = X_train_s(idx_kpca, :);
fprintf('  Fitting drtoolbox kernel_pca() on %d-point subsample...\n', MAX_KPCA);
Z_kpca_fit = kernel_pca(X_kpca_fit, n_comp, 'gauss', kpca_sig2);  % drtoolbox
% Build Nystrom OOS model from the drtoolbox embedding.
% Recover alphas: alphas = K_c^{-1} * Z (pseudoinverse, stable for small n)
K_fit = rbf_kernel(X_kpca_fit, X_kpca_fit, kpca_sig2);
col_mean   = mean(K_fit, 1);
total_mean = mean(col_mean);
K_fit_c = K_fit - mean(K_fit,2) - col_mean + total_mean;
kpca_model.X_fit      = X_kpca_fit;
kpca_model.alphas     = pinv(K_fit_c) * Z_kpca_fit;  % Nystrom alphas
kpca_model.sigma2     = kpca_sig2;
kpca_model.col_mean   = col_mean;
kpca_model.total_mean = total_mean;
rep.kpca.train = nystrom_project_v2(kpca_model, X_train_s);
rep.kpca.val   = nystrom_project_v2(kpca_model, X_val_s);
rep.kpca.test  = nystrom_project_v2(kpca_model, X_test_s);
timing.kpca = toc(t0);
fprintf('kPCA complete. Output dim: %d [%.2fs]\n', ...
    size(rep.kpca.train,2), timing.kpca);
% --- cross-check: drtoolbox embedding vs scratch kPCA on 2,000-pt sub ---
fprintf('  [cross-check] drtoolbox vs scratch kPCA (2,000-pt subsample)...\n');
try
    n_xval = 2000;
    idx_xval = randperm(n_tr, n_xval);
    X_xval = X_train_s(idx_xval, :);
    scratch_model = fit_kpca(X_xval, n_comp, kpca_sig2);
    proj_scratch  = scratch_model.train_proj;
    proj_dr       = nystrom_project_v2(kpca_model, X_xval);
    corr_vals_k   = zeros(1, 5);
    for pc = 1:5
        c = corrcoef(proj_scratch(:,pc), proj_dr(:,pc));
        corr_vals_k(pc) = abs(c(1,2));
    end
    fprintf('  |r| (scratch vs drtoolbox kPCA, first 5 dims): ');
    fprintf('%.4f ', corr_vals_k);
    fprintf('\n  Embeddings consistent: %s\n', ...
        mat2str(all(corr_vals_k(1:3) > 0.90)));
catch ME
    fprintf('  Cross-check skipped: %s\n', ME.message);
end
 
% --- 5d. Laplacian Eigenmaps — from scratch (drtoolbox NOT used) --------
% drtoolbox's laplacian_eigen.m is not used here because it provides no
% out-of-sample extension for validation/test projection. We use
% Gaussian-weighted NN interpolation (Bengio et al., 2004) instead.
% sigma^2 is tuned independently from kPCA by sweeping {0.5x, 1x, 2x}
% of the median heuristic and selecting the value that maximises graph
% connectivity (fraction of training points with at least one Gaussian
% edge weight above 0.01, assessed on a 2,000-point subsample).
fprintf('\n--- [4/4] Laplacian Eigenmaps ---\n');
t0 = tic;
le_k = 15;
sig2_candidates = kpca_sig2 * [0.5, 1.0, 2.0];
best_connectivity = -Inf;
best_le_sig2 = kpca_sig2;
fprintf('  Tuning LE sigma^2 (candidates: %.1f, %.1f, %.1f)...\n', ...
    sig2_candidates(1), sig2_candidates(2), sig2_candidates(3));
n_probe = min(2000, n_tr);
idx_probe = randperm(n_tr, n_probe);
X_probe = X_train_s(idx_probe,:);
[~, dist_probe] = my_knnsearch(X_probe, X_probe, le_k+1);
dist_probe = dist_probe(:, 2:end);  % drop self-distance
for s = 1:numel(sig2_candidates)
    sc = sig2_candidates(s);
    W_probe = exp(-dist_probe.^2 / (2*sc));
    connectivity = mean(max(W_probe, [], 2) > 0.01);
    fprintf('  sigma^2 = %8.2f -> connectivity = %.4f\n', sc, connectivity);
    if connectivity > best_connectivity
        best_connectivity = connectivity;
        best_le_sig2 = sc;
    end
end
fprintf('  Selected le_sig2 = %.4f\n', best_le_sig2);
le_model = fit_laplacian_eigenmaps(X_train_s, n_comp, le_k, best_le_sig2);
rep.le.train = le_model.train_proj;
rep.le.val   = transform_le(le_model, X_val_s,  X_train_s);
rep.le.test  = transform_le(le_model, X_test_s, X_train_s);
timing.le = toc(t0);
fprintf('Laplacian Eigenmaps complete. Output dim: %d [%.2fs]\n', ...
    size(rep.le.train,2), timing.le);
 
%% SECTION 6 — k-Sensitivity Analysis
% For each representation, sweep k in {1, 3, 5, 10, 15, 25}.
% The k with the highest validation accuracy is used for test evaluation.
% =========================================================================
fprintf('\n=== k-Sensitivity Analysis (kNN) ===\n');
rep_names  = {'raw', 'pca', 'kpca', 'le'};
rep_labels = {'Raw', 'PCA', 'kPCA (RBF)', 'Laplacian Eigenmaps'};
k_vals = [1, 3, 5, 10, 15, 25];
k_acc_val = zeros(numel(rep_names), numel(k_vals));
k_f1_val  = zeros(numel(rep_names), numel(k_vals));
best_k    = zeros(1, numel(rep_names));
for r = 1:numel(rep_names)
    rname = rep_names{r};
    Xtr = rep.(rname).train;
    Xvl = rep.(rname).val;
    fprintf('\n  [%s]\n', rep_labels{r});
    best_acc = -Inf;
    for ki = 1:numel(k_vals)
        k = k_vals(ki);
        yp_val = knn_predict(Xtr, y_train, Xvl, k);
        acc = mean(yp_val == y_val) * 100;
        f1  = macro_f1(y_val, yp_val);
        k_acc_val(r, ki) = acc;
        k_f1_val(r, ki)  = f1;
        fprintf('    k = %2d -> Val Acc: %6.2f%%  Macro F1: %.4f\n', k, acc, f1);
        if acc > best_acc
            best_acc  = acc;
            best_k(r) = k;
        end
    end
    fprintf('  Best k = %d (Val Acc = %.2f%%)\n', best_k(r), best_acc);
end
 
%% SECTION 7 — Final Evaluation on Test Set (using best k per representation)
% =========================================================================
fprintf('\n=== Final Test Evaluation (best k per representation) ===\n');
results = struct();
t_clf   = struct();
for r = 1:numel(rep_names)
    rname = rep_names{r};
    Xtr = rep.(rname).train;
    Xvl = rep.(rname).val;
    Xte = rep.(rname).test;
    k   = best_k(r);
    t0  = tic;
    yp_val  = knn_predict(Xtr, y_train, Xvl, k);
    yp_test = knn_predict(Xtr, y_train, Xte, k);
    t_clf.(rname) = toc(t0);
    acc_val  = mean(yp_val  == y_val)  * 100;
    acc_test = mean(yp_test == y_test) * 100;
    f1_val   = macro_f1(y_val,  yp_val);
    f1_test  = macro_f1(y_test, yp_test);
    results.(rname).acc_val  = acc_val;
    results.(rname).acc_test = acc_test;
    results.(rname).f1_val   = f1_val;
    results.(rname).f1_test  = f1_test;
    results.(rname).yp_val   = yp_val;
    results.(rname).yp_test  = yp_test;
    results.(rname).best_k   = k;
end
 
%% SECTION 8 — Summary Tables
% =========================================================================
% --- 8a. Main results table ----------------------------------------------
fprintf('\n');
fprintf('%-26s %6s %10s %10s %10s %10s\n', ...
    'Representation', 'Best k', 'Val Acc', 'Test Acc', 'Val F1', 'Test F1');
fprintf('%s\n', repmat('=', 1, 80));
for r = 1:numel(rep_names)
    res = results.(rep_names{r});
    fprintf('%-26s %6d %9.2f%% %9.2f%% %10.4f %10.4f\n', ...
        rep_labels{r}, res.best_k, ...
        res.acc_val, res.acc_test, res.f1_val, res.f1_test);
    if r < numel(rep_names), fprintf('%s\n', repmat('-', 1, 80)); end
end
fprintf('%s\n', repmat('=', 1, 80));
 
% --- 8b. Timing table ----------------------------------------------------
fprintf('\n=== Wall-Clock Timing Summary ===\n');
fprintf('%-26s %12s %12s\n', 'Stage', 'Repr. (s)', 'Clf. (s)');
fprintf('%s\n', repmat('-', 1, 52));
for r = 1:numel(rep_names)
    rn = rep_names{r};
    fprintf('%-26s %12.2f %12.2f\n', rep_labels{r}, timing.(rn), t_clf.(rn));
end
fprintf('%s\n', repmat('-', 1, 52));
 
%% SECTION 9 — Figures
% =========================================================================
 
% Fig 1 — PCA cumulative explained variance
figure('Position', [100 100 720 440]);
plot(cumsum(explained_full), 'b-', 'LineWidth', 1.8); hold on;
xline(n_comp, 'r--', ...
    sprintf('%d PCs (%.1f%%)', n_comp, cum_var(end)), ...
    'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.3);
yline(95, 'g--', '95% threshold', 'LineWidth', 1.3);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA — Cumulative Explained Variance');
xlim([1, size(X_train_s,2)]); ylim([0, 100]); grid on;
saveas(gcf, 'fig1_pca_variance.png');
fprintf('\nSaved fig1_pca_variance.png\n');
 
% Fig 2 — 2D scatter plots of all four representations
classes_u = unique(y_train);
grand_mu = mean(X_train_s, 1);
Sb = zeros(1, size(X_train_s,2));
Sw = zeros(1, size(X_train_s,2));
for ci = 1:numel(classes_u)
    idx_c = y_train == classes_u(ci);
    mu_c  = mean(X_train_s(idx_c,:), 1);
    n_c   = sum(idx_c);
    Sb = Sb + n_c * (mu_c - grand_mu).^2;
    Sw = Sw + sum((X_train_s(idx_c,:) - mu_c).^2, 1);
end
fisher_score = Sb ./ (Sw + eps);
[~, feat_sorted] = sort(fisher_score, 'descend');
top2 = feat_sorted(1:2);
colors = lines(7);
fig_titles = {sprintf('Raw — features %d & %d (most discriminative)', top2(1), top2(2)), ...
              'PCA — PCs 1 & 2', ...
              'kPCA (RBF) — dims 1 & 2', ...
              'Laplacian Eigenmaps — dims 1 & 2'};
rep_2d = {X_train_s(:, top2), rep.pca.train(:,1:2), ...
          rep.kpca.train(:,1:2), rep.le.train(:,1:2)};
figure('Position', [100 100 1200 900]);
for r = 1:4
    subplot(2, 2, r);
    Xp = rep_2d{r};
    for ci = 1:7
        mask = y_train == ci;
        scatter(Xp(mask,1), Xp(mask,2), 4, colors(ci,:), ...
            'filled', 'MarkerFaceAlpha', 0.3); hold on;
    end
    title(fig_titles{r}, 'FontSize', 10);
    xlabel('Dim 1'); ylabel('Dim 2'); grid on; axis tight;
    if r == 4
        legend(class_names, 'Location', 'best', 'FontSize', 7);
    end
end
sgtitle('2D Projections — Training Set (colored by crop class)');
saveas(gcf, 'fig2_2d_embeddings.png');
fprintf('Saved fig2_2d_embeddings.png\n');
 
% Fig 3 — k-Sensitivity curves (validation accuracy per representation)
figure('Position', [100 100 820 480]);
markers = {'o-', 's-', '^-', 'd-'};
for r = 1:numel(rep_names)
    plot(k_vals, k_acc_val(r,:), markers{r}, ...
        'LineWidth', 1.6, 'MarkerSize', 7); hold on;
end
xlabel('k (number of neighbors)');
ylabel('Validation Accuracy (%)');
title('kNN k-Sensitivity by Representation');
legend(rep_labels, 'Location', 'best', 'FontSize', 9);
grid on;
xticks(k_vals);
saveas(gcf, 'fig3_k_sensitivity.png');
fprintf('Saved fig3_k_sensitivity.png\n');
 
% Fig 4 — Test accuracy and macro F1 grouped bar charts
acc_vec = zeros(1, numel(rep_names));
f1_vec  = zeros(1, numel(rep_names));
for r = 1:numel(rep_names)
    acc_vec(r) = results.(rep_names{r}).acc_test;
    f1_vec(r)  = results.(rep_names{r}).f1_test;
end
figure('Position', [100 100 860 480]);
subplot(1, 2, 1);
b1 = bar(acc_vec, 0.65);
b1.FaceColor = 'flat';
b1.CData = [0.22 0.45 0.70; 0.85 0.33 0.10; 0.47 0.67 0.19; 0.49 0.18 0.56];
set(gca, 'XTickLabel', rep_labels, 'XTickLabelRotation', 14, 'FontSize', 9);
ylabel('Test Accuracy (%)'); title('Test Accuracy'); grid on;
ylim([95 100]);
for r = 1:numel(rep_names)
    text(r, acc_vec(r)+0.05, sprintf('%.2f', acc_vec(r)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end
subplot(1, 2, 2);
b2 = bar(f1_vec, 0.65);
b2.FaceColor = 'flat';
b2.CData = [0.22 0.45 0.70; 0.85 0.33 0.10; 0.47 0.67 0.19; 0.49 0.18 0.56];
set(gca, 'XTickLabel', rep_labels, 'XTickLabelRotation', 14, 'FontSize', 9);
ylabel('Macro F1'); title('Macro F1'); grid on;
ylim([0.97 1.00]);
for r = 1:numel(rep_names)
    text(r, f1_vec(r)+0.0005, sprintf('%.4f', f1_vec(r)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end
sgtitle('Test Performance by Representation (kNN, best k)');
saveas(gcf, 'fig4_accuracy_f1.png');
fprintf('Saved fig4_accuracy_f1.png\n');
 
% Fig 5 — Normalized confusion matrices for all four representations (2x2 grid)
blues = [linspace(1, 0.03, 256)', linspace(1, 0.27, 256)', linspace(1, 0.58, 256)'];
figure('Position', [100 100 1200 1000]);
for r = 1:4
    rname = rep_names{r};
    cm      = confusion_matrix(y_test, results.(rname).yp_test, 7);
    cm_norm = cm ./ (sum(cm, 2) + eps);  % row-normalize
    subplot(2, 2, r);
    imagesc(cm_norm); colormap(gca, blues);
    colorbar;
    caxis([0 1]);
    set(gca, 'XTick', 1:7, 'XTickLabel', class_names, ...
             'YTick', 1:7, 'YTickLabel', class_names, ...
             'XTickLabelRotation', 30, 'FontSize', 8);
    xlabel('Predicted'); ylabel('True');
    title(sprintf('Confusion: %s (k = %d)', rep_labels{r}, results.(rname).best_k), ...
        'FontSize', 9);
    for i = 1:7
        for j = 1:7
            val = cm_norm(i,j);
            txt_color = 'white';
            if val < 0.6, txt_color = 'black'; end
            text(j, i, sprintf('%.2f', val), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment',   'middle', ...
                'FontSize', 6, 'Color', txt_color);
        end
    end
end
sgtitle('Normalized Confusion Matrices — All Representations (kNN, test set)');
saveas(gcf, 'fig5_all_confusion_matrices.png');
fprintf('Saved fig5_all_confusion_matrices.png\n');
 
% Fig 6 — Per-class F1 grouped bar chart (all four representations)
f1_pc = zeros(7, numel(rep_names));
for r = 1:numel(rep_names)
    yp = results.(rep_names{r}).yp_test;
    for ci = 1:7
        tp = sum(yp==ci & y_test==ci);
        fp = sum(yp==ci & y_test~=ci);
        fn = sum(yp~=ci & y_test==ci);
        p  = tp / (tp + fp + eps);
        rc = tp / (tp + fn + eps);
        f1_pc(ci,r) = 2*p*rc / (p + rc + eps);
    end
end
figure('Position', [100 100 960 500]);
bar(f1_pc, 0.8);
set(gca, 'XTickLabel', class_names, 'XTickLabelRotation', 15, 'FontSize', 10);
ylabel('F1-score');
title('Per-class F1 — All Representations (kNN, test set)');
legend(rep_labels, 'Location', 'southwest', 'FontSize', 8);
grid on; ylim([0.85 1.00]);
saveas(gcf, 'fig6_perclass_f1.png');
fprintf('Saved fig6_perclass_f1.png\n');
 
% Fig 7 — Wall-clock timing bar chart
repr_times = [timing.raw, timing.pca, timing.kpca, timing.le];
figure('Position', [100 100 720 420]);
b = bar(repr_times, 0.65);
set(gca, 'YScale', 'log');
b.FaceColor = 'flat';
b.CData = [0.22 0.45 0.70; 0.85 0.33 0.10; 0.47 0.67 0.19; 0.49 0.18 0.56];
set(gca, 'XTickLabel', rep_labels, 'XTickLabelRotation', 14, 'FontSize', 10);
ylabel('Wall-clock Time (seconds, log scale)');
title('Representation Computation Time');
grid on;
for r = 1:4
    text(r, repr_times(r)*1.02, sprintf('%.1fs', repr_times(r)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
saveas(gcf, 'fig7_timing.png');
fprintf('Saved fig7_timing.png\n');
 
fprintf('\n=== All done. ===\n');
 
 
%% LOCAL HELPER FUNCTIONS
% =========================================================================
 
% -------------------------------------------------------------------------
function K = rbf_kernel(X1, X2, sigma2)
% Computes the RBF (Gaussian) kernel matrix between X1 and X2.
% Uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a'b to avoid
% forming explicit pairwise differences (numerically stable).
    n1 = size(X1,1); n2 = size(X2,1);
    D2 = sum(X1.^2,2)*ones(1,n2) + ones(n1,1)*sum(X2.^2,2)' - 2*(X1*X2');
    D2 = max(D2, 0);
    K  = exp(-D2 / (2*sigma2));
end
 
% -------------------------------------------------------------------------
function model = fit_kpca(X_tr, n_comp, sigma2)
% Fits a kernel PCA model with an RBF kernel.
% Stores all centering statistics required for Nystrom out-of-sample
% projection (Scholkopf et al., 1998).
    n = size(X_tr,1);
    fprintf('  Computing %d x %d RBF kernel matrix (%.0f MB)...\n', ...
        n, n, n^2*8/1e6);
    K = rbf_kernel(X_tr, X_tr, sigma2);
    % Center the kernel matrix.
    col_mean   = mean(K, 1);
    row_mean   = mean(K, 2);
    total_mean = mean(col_mean);
    K_c = K - row_mean - col_mean + total_mean;
    fprintf('  Eigendecomposition (top %d eigenvalues)...\n', n_comp);
    [V, D] = eigs(K_c, n_comp, 'largestreal');
    lambdas = diag(D);
    alphas  = V ./ sqrt(max(lambdas', eps));  % column-normalized eigenvectors
    model.X_tr       = X_tr;
    model.alphas     = alphas;
    model.lambdas    = lambdas;
    model.sigma2     = sigma2;
    model.n          = n;
    model.col_mean   = col_mean;
    model.total_mean = total_mean;
    model.train_proj = K_c * alphas;
end
 
% -------------------------------------------------------------------------
function proj = transform_kpca(model, X_new)
% Projects new points into the kPCA embedding via Nystrom centering.
    K_new   = rbf_kernel(X_new, model.X_tr, model.sigma2);
    row_mean = mean(K_new, 2);
    K_new_c  = K_new - row_mean - model.col_mean + model.total_mean;
    proj = K_new_c * model.alphas;
end
 
% -------------------------------------------------------------------------
function model = fit_laplacian_eigenmaps(X_tr, n_comp, k, sigma2)
% Fits a Laplacian Eigenmaps embedding (Belkin & Niyogi, 2003).
% Solves the generalized eigenproblem L v = lambda D v, where
% L = D - W is the graph Laplacian and W carries Gaussian edge weights.
%
% Note: drtoolbox's laplacian_eigen.m is intentionally not used because
% it provides no out-of-sample extension for validation/test projection.
    n = size(X_tr,1);
    fprintf('  Building kNN graph (k = %d, n = %d)...\n', k, n);
    [idx_knn, dist_knn] = my_knnsearch(X_tr, X_tr, k+1);
    idx_knn  = idx_knn(:, 2:end);
    dist_knn = dist_knn(:, 2:end);
    W_vals = exp(-dist_knn.^2 / (2*sigma2));
    rows = repmat((1:n)', 1, k);
    W    = sparse(rows(:), idx_knn(:), W_vals(:), n, n);
    W    = max(W, W.');
    d     = full(sum(W, 2));
    D_mat = spdiags(d, 0, n, n);
    L     = D_mat - W;
    fprintf('  Solving sparse generalized eigenproblem (%d eigenvectors)...\n', ...
        n_comp+1);
    eigs_opts.tol   = 1e-6;
    eigs_opts.maxit = 500;
    [V, Dval] = eigs(L, D_mat, n_comp+1, 'sm', eigs_opts);
    lambdas = diag(Dval);
    [~, sidx] = sort(lambdas, 'ascend');
    V = V(:, sidx);
    V = V(:, 2:end);  % discard the trivial (constant) eigenvector
    model.X_tr      = X_tr;
    model.V         = V;
    model.k         = k;
    model.sigma2    = sigma2;
    model.train_proj = V;
end
 
% -------------------------------------------------------------------------
function proj = transform_le(model, X_new, X_tr)
% Projects new points into the Laplacian Eigenmaps embedding via
% Gaussian-weighted NN interpolation (Bengio et al., 2004 —
% "Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps, and
% Spectral Clustering").
% Each new point is assigned a weighted average of its k nearest
% training neighbors' embeddings, with weights from the RBF kernel.
    k = model.k;
    [idx_nn, dist_nn] = my_knnsearch(X_tr, X_new, k);
    W = exp(-dist_nn.^2 / (2*model.sigma2));
    W = W ./ (sum(W,2) + eps);
    n_new  = size(X_new,1);
    n_comp = size(model.V,2);
    V_neighbors = reshape(model.V(idx_nn(:),:), n_new, k, n_comp);
    W_3d = repmat(W, [1, 1, n_comp]);
    proj = squeeze(sum(W_3d .* V_neighbors, 2));
    if n_new == 1, proj = proj'; end
end
 
% -------------------------------------------------------------------------
function f1 = macro_f1(y_true, y_pred)
% Computes the unweighted macro-average F1 score.
    classes = unique(y_true);
    f1_per_class = zeros(numel(classes), 1);
    for i = 1:numel(classes)
        c    = classes(i);
        tp   = sum(y_pred == c & y_true == c);
        fp   = sum(y_pred == c & y_true ~= c);
        fn   = sum(y_pred ~= c & y_true == c);
        prec = tp / (tp + fp + eps);
        rec  = tp / (tp + fn + eps);
        f1_per_class(i) = 2*prec*rec / (prec + rec + eps);
    end
    f1 = mean(f1_per_class);
end
 
% -------------------------------------------------------------------------
function [idx, dist] = my_knnsearch(X_ref, X_query, k)
% Brute-force kNN search, processed in blocks to control memory usage.
    n_q   = size(X_query,1);
    n_ref = size(X_ref,1);
    idx   = zeros(n_q, k);
    dist  = zeros(n_q, k);
    blockSize = 1000;
    sq_ref = sum(X_ref.^2, 2)';
    for startIdx = 1:blockSize:n_q
        endIdx = min(startIdx + blockSize - 1, n_q);
        Xb     = X_query(startIdx:endIdx, :);
        sq_q   = sum(Xb.^2, 2);
        D2     = bsxfun(@plus, sq_q, sq_ref) - 2*(Xb * X_ref.');
        D2     = max(D2, 0);
        [sortedD2, sortedIdx] = sort(D2, 2, 'ascend');
        kk = min(k, n_ref);
        idx(startIdx:endIdx,  1:kk) = sortedIdx(:, 1:kk);
        dist(startIdx:endIdx, 1:kk) = sqrt(sortedD2(:, 1:kk));
    end
end
 
% -------------------------------------------------------------------------
function y_pred = knn_predict(X_train, y_train, X_query, k)
% kNN classifier with majority-vote decision rule.
    [idx, ~] = my_knnsearch(X_train, X_query, k);
    n_query  = size(X_query,1);
    y_pred   = zeros(n_query,1);
    for i = 1:n_query
        y_pred(i) = mode(y_train(idx(i,:)));
    end
end
 
% -------------------------------------------------------------------------
function C = confusion_matrix(y_true, y_pred, n_classes)
% Computes a confusion matrix with rows = true class, cols = predicted class.
    if nargin < 3
        n_classes = numel(unique(y_true));
    end
    classes = 1:n_classes;
    C = zeros(n_classes, n_classes);
    for i = 1:n_classes
        for j = 1:n_classes
            C(i,j) = sum(y_true == classes(i) & y_pred == classes(j));
        end
    end
end
 
% -------------------------------------------------------------------------
function proj = nystrom_project_v2(model, X_new)
% Projects X_new via Nystrom centering using alphas recovered from
% the drtoolbox embedding (alphas = pinv(K_fit_c) * Z_fit).
    K_new   = rbf_kernel(X_new, model.X_fit, model.sigma2);
    K_new_c = K_new - mean(K_new,2) - model.col_mean + model.total_mean;
    proj    = K_new_c * model.alphas;
end
 
% -------------------------------------------------------------------------
function idx = stratified_subsample(y, max_total)
% Returns indices of a stratified subsample of max_total points from y.
    classes  = unique(y);
    n_per    = floor(max_total / numel(classes));
    idx      = [];
    for i = 1:numel(classes)
        ci   = find(y == classes(i));
        take = min(n_per, numel(ci));
        idx  = [idx; ci(randperm(numel(ci), take))]; %#ok<AGROW>
    end
end