#!/usr/bin/env Rscript
# Drive proDA reference on TSV inputs dumped by the Python side.
#
# Usage:
#   Rscript r_reference_driver.R <M_tsv> <design_tsv> <out_dir>
#
# Outputs (in out_dir):
#   coefs.tsv         per-protein coefficient matrix from proDA()
#   s2.tsv            per-protein s2 (variance estimate)
#   df.tsv            per-protein df (degrees of freedom)
#   rho.tsv           per-sample dropout-curve position
#   zeta.tsv          per-sample dropout-curve scale
#   invprobit.tsv     invprobit(x, rho, zeta) on a fixed test grid
#   test_diff.tsv     test_diff() Wald-test result (contrast = group2 - group1)
#   timing.tsv        wall-clock per step (proda fit / test_diff)

suppressPackageStartupMessages({
  library(proDA)
})

args <- commandArgs(trailingOnly = TRUE)
M_tsv      <- args[[1]]
design_tsv <- args[[2]]
out_dir    <- args[[3]]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

M <- as.matrix(read.table(M_tsv, sep = "\t", header = TRUE, row.names = 1,
                          check.names = FALSE, na.strings = c("NA","")))
design_df <- read.table(design_tsv, sep = "\t", header = TRUE, row.names = 1,
                        check.names = FALSE)
# Use the same dummy encoding as Python: intercept + (groups-1) dummies.
# The Python side passes a single column "group" with values "A"/"B".
group_vec <- design_df[, 1]
design_mat <- model.matrix(~ group_vec)
colnames(design_mat) <- c("Intercept", paste0("group", levels(factor(group_vec))[-1]))
rownames(design_mat) <- colnames(M)

# 1) invprobit on a fixed grid (deterministic, bit-exact target).
grid_x   <- seq(15, 22, length.out = 50)
grid_rho <- 18.0
grid_zeta <- 0.5
inv_vec <- proDA::invprobit(grid_x, rho = grid_rho, zeta = grid_zeta)
write.table(data.frame(x = grid_x, invprobit = inv_vec),
            file.path(out_dir, "invprobit.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# 2) Fit proDA (NO moderation — to match v0.1 Python).
t0 <- proc.time()[[3]]
fit <- proDA(M, design = design_mat,
             moderate_location = FALSE,
             moderate_variance = FALSE,
             verbose = FALSE)
t1 <- proc.time()[[3]]

# Coefs / s2 / df / rho / zeta.
write.table(coefficients(fit), file.path(out_dir, "coefs.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
feat <- feature_parameters(fit)
write.table(data.frame(name = rownames(coefficients(fit)),
                       s2 = feat$s2, df = feat$df, n_obs = feat$n_obs),
            file.path(out_dir, "s2_df.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
hp <- hyper_parameters(fit)
write.table(data.frame(sample = colnames(M),
                       rho = hp$dropout_curve_position,
                       zeta = hp$dropout_curve_scale),
            file.path(out_dir, "rho_zeta.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# 3) test_diff on the group contrast (B - A; intercept stays).
t2 <- proc.time()[[3]]
res <- test_diff(fit, contrast = "groupB")
t3 <- proc.time()[[3]]
write.table(res, file.path(out_dir, "test_diff.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

write.table(data.frame(proda_fit = t1 - t0, test_diff = t3 - t2),
            file.path(out_dir, "timing.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# 4) pd_row_t_test reference (un-moderated, to compare t-statistic).
n1 <- sum(group_vec == levels(factor(group_vec))[1])
Xmat <- M[, group_vec == levels(factor(group_vec))[1], drop = FALSE]
Ymat <- M[, group_vec == levels(factor(group_vec))[2], drop = FALSE]
tt <- proDA::pd_row_t_test(Xmat, Ymat,
                           moderate_location = FALSE,
                           moderate_variance = FALSE)
write.table(data.frame(name = tt$name, t_statistic = tt$t_statistic,
                       pval = tt$pval, diff = tt$diff),
            file.path(out_dir, "pd_row_t_test.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# 5) moderate_variance reference — per-protein s2 with EB shrinkage on.
fit_mod <- proDA(M, design = design_mat,
                 moderate_location = FALSE,
                 moderate_variance = TRUE,
                 verbose = FALSE)
feat_mod <- feature_parameters(fit_mod)
write.table(data.frame(name = rownames(coefficients(fit_mod)),
                       s2 = feat_mod$s2, df = feat_mod$df),
            file.path(out_dir, "s2_moderated.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# 6) dist_approx reference (between proteins, blinded model).
da <- proDA::dist_approx(fit, by_sample = FALSE)
dm <- as.matrix(da$mean)
write.table(dm, file.path(out_dir, "dist_approx.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

cat("R proDA pipeline done\n")
