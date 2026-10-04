#!/usr/bin/env Rscript
# H2 (docs/experiments/E01_design.md): does temperature-only sampling (arm "topp1") have a larger
# seed (sampling) variance than the Qwen3 recommended sampling (arm "qwen")?
# For each experiment x precision x prefix-cache with both arms on the same questions:
#   primary: difference var_seed(topp1) - var_seed(qwen) (stable when the qwen variance is ~0);
#   secondary: ratio (reported only when var_seed(qwen) > 0); and the accuracy difference,
# with 95% intervals from a joint question-level bootstrap (the same resampled questions are used
# for both arms, so the interval reflects the paired design).
#
# Input : runs/_derived/runs_flat.csv
# Output: results/05_h2_seed_variance.csv

INPUT <- "runs/_derived/runs_flat.csv"
ARM_A <- "qwen"
ARM_B <- "topp1"
B <- 1000

if (!file.exists(INPUT)) {
  message("05_h2_seed_variance: ", INPUT, " not found, skipping")
  quit(status = 0)
}
runs <- read.csv(INPUT, stringsAsFactors = FALSE)
if (!"arm" %in% names(runs) || !all(c(ARM_A, ARM_B) %in% runs$arm)) {
  message("05_h2_seed_variance: arms '", ARM_A, "' and '", ARM_B, "' not both present, skipping")
  quit(status = 0)
}
if (!"precision" %in% names(runs)) runs$precision <- ""

# Seed variance by method of moments: spread of seed means within a question, minus the part
# explained by same-seed replicate noise (balanced design: R replicates per seed).
seed_variance <- function(y, q, s) {
  cell <- paste(q, s)
  cell_mean <- tapply(y, cell, mean)
  cell_q <- tapply(q, cell, function(x) x[1])
  r <- mean(tapply(y, cell, length))
  within <- tapply(y, cell, function(x) if (length(x) > 1) var(x) else NA)
  v_rep <- if (all(is.na(within))) 0 else mean(within, na.rm = TRUE)
  between <- tapply(cell_mean, cell_q, function(x) if (length(x) > 1) var(x) else NA)
  max(0, mean(between, na.rm = TRUE) - v_rep / r)
}

stats_for <- function(a, b) {
  va <- seed_variance(a$em, a$question_id, a$seed_index)
  vb <- seed_variance(b$em, b$question_id, b$seed_index)
  c(var_seed_a = va, var_seed_b = vb, diff = vb - va, ratio = if (va > 0) vb / va else NA,
    acc_a = mean(a$em), acc_b = mean(b$em), acc_diff = mean(b$em) - mean(a$em))
}

set.seed(20261003)
out <- list()
groups <- unique(runs[c("experiment", "precision", "prefix_cache")])
for (g in seq_len(nrow(groups))) {
  key <- groups[g, ]
  sel <- runs$experiment == key$experiment & runs$precision == key$precision &
    runs$prefix_cache == key$prefix_cache
  a <- runs[sel & runs$arm == ARM_A, ]
  b <- runs[sel & runs$arm == ARM_B, ]
  shared <- intersect(unique(a$question_id), unique(b$question_id))
  if (length(shared) < 10) next
  a <- a[a$question_id %in% shared, ]
  b <- b[b$question_id %in% shared, ]
  point <- stats_for(a, b)
  ia <- split(seq_len(nrow(a)), a$question_id)
  ib <- split(seq_len(nrow(b)), b$question_id)
  boot <- replicate(B, {
    pick <- sample(shared, length(shared), replace = TRUE)
    relabel <- function(d, idx) {
      rows <- unlist(idx[pick], use.names = FALSE)
      d <- d[rows, ]
      d$question_id <- rep(seq_along(pick), times = lengths(idx[pick]))
      d
    }
    stats_for(relabel(a, ia), relabel(b, ib))
  })
  ci <- apply(boot, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE)
  out[[length(out) + 1]] <- data.frame(
    experiment = key$experiment, precision = key$precision, prefix_cache = key$prefix_cache,
    n_questions = length(shared),
    var_seed_qwen = point[["var_seed_a"]], var_seed_topp1 = point[["var_seed_b"]],
    var_seed_diff = point[["diff"]], diff_lo = ci[1, "diff"], diff_hi = ci[2, "diff"],
    ratio = point[["ratio"]],
    acc_qwen = point[["acc_a"]], acc_topp1 = point[["acc_b"]],
    acc_diff = point[["acc_diff"]], acc_diff_lo = ci[1, "acc_diff"], acc_diff_hi = ci[2, "acc_diff"]
  )
}
if (!length(out)) {
  message("05_h2_seed_variance: no comparable groups, skipping")
  quit(status = 0)
}
dir.create("results", showWarnings = FALSE)
write.csv(do.call(rbind, out), "results/05_h2_seed_variance.csv", row.names = FALSE)
message(sprintf("05_h2_seed_variance: %d groups", length(out)))
