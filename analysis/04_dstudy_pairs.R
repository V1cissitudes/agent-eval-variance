#!/usr/bin/env Rscript
# D-study for concrete comparisons (docs/experiments/E01_design.md, Q2): for every pair of
# conditions that share questions and differ in exactly one factor (precision, arm, temperature,
# prefix cache, thinking), estimate the question x condition interaction variance from the
# per-question differences and project SE, power (3 pp) and ranking-flip probability over
# (n_q, n_r). 95% intervals come from a question-level bootstrap.
#
# Input : runs/_derived/runs_flat.csv
# Output: results/04_dstudy_pairs.csv, figures/04_<experiment>_dstudy_pairs.png
#
# Paired decomposition (method of moments). Both conditions use the same questions and the same seeds
# (seed_index), so the comparison is built from same-question, same-seed differences:
#   m_k(q,s) = mean over the R same-seed replicates in condition k, w_k(q,s) = their variance
#   v_rep_k  = mean of w_k(q,s)                                      (serving noise, condition k)
#   d(q,s)   = m_1(q,s) - m_2(q,s);  d(q) = mean_s d(q,s)
#   v_ds     = mean_q var_s d(q,s) - (v_rep_1 + v_rep_2) / R         (seed-level difference variance)
#   v_dq     = var_q d(q) - v_ds / S - (v_rep_1 + v_rep_2) / (S R)   (question x condition variance)
# With one seed per question (greedy) v_ds is not identifiable and is set to 0 (no sampling at T = 0).
# Projection for a new study with n_q questions and n_r independent runs per question and condition
# (a fresh seed each, one replicate), paired by question and seed:
#   SE_diff^2 = v_dq / n_q + (v_ds + v_rep_1 + v_rep_2) / (n_q n_r)
# Power is two-sided at ALPHA: P = Phi(D/SE - z) + Phi(-D/SE - z); ranking flip is Phi(-D/SE).
# Pairing by seed only helps when both conditions really are run with the same seeds (as here). For
# conditions whose sampling is not linked by the seed (e.g. two different models), the unpaired
# projection uses each condition's own seed + replicate variance instead of v_ds:
#   SE_unpaired^2 = v_dq / n_q + (v_seed_1 + v_rep_1 + v_seed_2 + v_rep_2) / (n_q n_r)

INPUT <- "runs/_derived/runs_flat.csv"
DELTA <- 0.03
ALPHA <- 0.05
B <- 500
N_Q <- c(50, 100, 200, 300, 500, 1000)
N_R <- c(1, 2, 3, 5, 10)
FACTORS <- c("precision", "arm", "temperature", "prefix_cache", "thinking")

if (!file.exists(INPUT)) {
  message("04_dstudy_pairs: ", INPUT, " not found, skipping")
  quit(status = 0)
}
runs <- read.csv(INPUT, stringsAsFactors = FALSE)
for (f in FACTORS) if (!f %in% names(runs)) runs[[f]] <- ""
# The precision comparison spans two experiments: BF16 comes from E01a (arms greedy, qwen) and
# FP8 / INT4 from E01b, run with the same questions and seeds. Pool them as one comparison family.
if (all(c("E01a", "E01b") %in% runs$experiment)) {
  pooled <- runs[(runs$experiment == "E01a" & runs$arm %in% c("greedy", "qwen")) |
                   runs$experiment == "E01b", ]
  pooled$experiment <- "E01_precision"
  runs <- rbind(runs, pooled)
}
runs$cond <- do.call(paste, c(runs[FACTORS], sep = "|"))

# cells (question, seed) of one condition: replicate mean, replicate variance and count
cells_of <- function(d) {
  key <- paste(d$question_id, d$seed_index, sep = "\r")
  data.frame(
    cell = names(tapply(d$em, key, mean)),
    q = sub("\r.*", "", names(tapply(d$em, key, mean))),
    m = as.vector(tapply(d$em, key, mean)),
    w = as.vector(tapply(d$em, key, function(x) if (length(x) > 1) var(x) else NA)),
    r = as.vector(tapply(d$em, key, length))
  )
}

pair_estimates <- function(a, b) {
  m <- merge(a, b, by = c("cell", "q"), suffixes = c("1", "2"))
  r <- mean(c(m$r1, m$r2))
  v_rep1 <- if (all(is.na(m$w1))) 0 else mean(m$w1, na.rm = TRUE)
  v_rep2 <- if (all(is.na(m$w2))) 0 else mean(m$w2, na.rm = TRUE)
  m$d <- m$m1 - m$m2
  s <- mean(tapply(m$cell, m$q, length))
  dq <- tapply(m$d, m$q, mean)
  v_ds <- if (s >= 2) {
    max(0, mean(tapply(m$d, m$q, var), na.rm = TRUE) - (v_rep1 + v_rep2) / r)
  } else 0
  v_dq <- max(0, var(dq) - v_ds / s - (v_rep1 + v_rep2) / (s * r))
  seed_var <- function(mk, v_rep) {  # condition's own seed variance (0 with one seed per question)
    if (s < 2) return(0)
    max(0, mean(tapply(mk, m$q, var), na.rm = TRUE) - v_rep / r)
  }
  c(diff = mean(m$d), v_dq = v_dq, v_ds = v_ds, v_rep1 = v_rep1, v_rep2 = v_rep2,
    v_seed1 = seed_var(m$m1, v_rep1), v_seed2 = seed_var(m$m2, v_rep2),
    n_q_obs = length(dq), seeds = s, replicates = r)
}

project <- function(est) {
  g <- expand.grid(n_q = N_Q, n_r = N_R)
  se <- sqrt(est[["v_dq"]] / g$n_q +
               (est[["v_ds"]] + est[["v_rep1"]] + est[["v_rep2"]]) / (g$n_q * g$n_r))
  z <- qnorm(1 - ALPHA / 2)
  power <- function(x) ifelse(x > 0, pnorm(DELTA / x - z) + pnorm(-DELTA / x - z), 1)
  flip <- function(x) ifelse(x > 0, pnorm(-DELTA / x), 0)
  g$se_difference <- se
  g$power <- power(se)
  g$p_rank_flip <- flip(se)
  se_u <- sqrt(est[["v_dq"]] / g$n_q + (est[["v_seed1"]] + est[["v_rep1"]] + est[["v_seed2"]] +
                                          est[["v_rep2"]]) / (g$n_q * g$n_r))
  g$se_difference_unpaired <- se_u
  g$power_unpaired <- power(se_u)
  g$p_rank_flip_unpaired <- flip(se_u)
  g
}

differs_in_one_factor <- function(c1, c2) {
  sum(strsplit(c1, "|", fixed = TRUE)[[1]] != strsplit(c2, "|", fixed = TRUE)[[1]]) == 1
}

set.seed(20261003)
out <- list()
for (exp in unique(runs$experiment)) {
  de <- runs[runs$experiment == exp, ]
  conds <- sort(unique(de$cond))
  pq <- lapply(setNames(conds, conds), function(c) cells_of(de[de$cond == c, ]))
  for (i in seq_along(conds)) for (j in seq_along(conds)) {
    if (j <= i || !differs_in_one_factor(conds[i], conds[j])) next
    a <- pq[[conds[i]]]
    b <- pq[[conds[j]]]
    est <- pair_estimates(a, b)
    point <- project(est)
    shared <- intersect(a$q, b$q)
    ia <- split(seq_len(nrow(a)), a$q)
    ib <- split(seq_len(nrow(b)), b$q)
    relabel <- function(x, idx, pick) {  # whole questions, renumbered so duplicates stay distinct
      rows <- unlist(idx[pick], use.names = FALSE)
      x <- x[rows, ]
      newq <- rep(seq_along(pick), times = lengths(idx[pick]))
      x$q <- as.character(newq)
      x$cell <- paste(newq, sub(".*\r", "", x$cell), sep = "\r")
      x
    }
    boot <- replicate(B, {
      pick <- sample(shared, length(shared), replace = TRUE)
      pr <- project(pair_estimates(relabel(a, ia, pick), relabel(b, ib, pick)))
      c(pr$power, pr$power_unpaired)
    })
    k <- nrow(point)
    point$power_lo <- apply(boot[1:k, , drop = FALSE], 1, quantile, 0.025)
    point$power_hi <- apply(boot[1:k, , drop = FALSE], 1, quantile, 0.975)
    point$power_unpaired_lo <- apply(boot[k + 1:k, , drop = FALSE], 1, quantile, 0.025)
    point$power_unpaired_hi <- apply(boot[k + 1:k, , drop = FALSE], 1, quantile, 0.975)
    out[[length(out) + 1]] <- cbind(
      experiment = exp, condition_1 = conds[i], condition_2 = conds[j],
      observed_diff = est[["diff"]], var_question_x_condition = est[["v_dq"]],
      var_seed_difference = est[["v_ds"]], var_replicate_1 = est[["v_rep1"]],
      var_replicate_2 = est[["v_rep2"]], var_seed_1 = est[["v_seed1"]], var_seed_2 = est[["v_seed2"]],
      seeds = est[["seeds"]], replicates = est[["replicates"]], point
    )
  }
}
if (!length(out)) {
  message("04_dstudy_pairs: no comparable condition pairs, skipping")
  quit(status = 0)
}
res <- do.call(rbind, out)
dir.create("results", showWarnings = FALSE)
write.csv(res, "results/04_dstudy_pairs.csv", row.names = FALSE)

dir.create("figures", showWarnings = FALSE)
for (exp in unique(res$experiment)) {
  rr <- res[res$experiment == exp, ]
  pairs <- unique(rr[c("condition_1", "condition_2")])
  ncol <- min(3, nrow(pairs))
  nrow_ <- ceiling(nrow(pairs) / ncol)
  # type = "cairo": no EXIF/ICC chunks (the macOS quartz default adds them)
  png(sprintf("figures/04_%s_dstudy_pairs.png", exp), width = 560 * ncol, height = 420 * nrow_,
      res = 110, type = "cairo")
  par(mfrow = c(nrow_, ncol), mar = c(4.2, 4.2, 3.2, 1))
  cols <- hcl.colors(length(N_R), "Viridis")
  for (k in seq_len(nrow(pairs))) {
    pk <- rr[rr$condition_1 == pairs$condition_1[k] & rr$condition_2 == pairs$condition_2[k], ]
    title <- sprintf("%s\nvs %s", pairs$condition_1[k], pairs$condition_2[k])
    plot(NA, xlim = range(N_Q), ylim = c(0, 1), log = "x", xlab = "questions (n_q)",
         ylab = sprintf("power to detect %.0f pp", 100 * DELTA), main = title, cex.main = 0.7)
    abline(h = 0.8, lty = 2, col = "grey50")
    for (r in seq_along(N_R)) {
      pr <- pk[pk$n_r == N_R[r], ]
      polygon(c(pr$n_q, rev(pr$n_q)), c(pr$power_lo, rev(pr$power_hi)),
              col = adjustcolor(cols[r], 0.15), border = NA)
      lines(pr$n_q, pr$power, col = cols[r], lwd = 2)
    }
    legend("bottomright", legend = paste("n_r =", N_R), col = cols, lwd = 2, cex = 0.65, bty = "n")
  }
  dev.off()
}
message(sprintf("04_dstudy_pairs: %d comparisons", nrow(unique(res[c("experiment", "condition_1",
                                                                       "condition_2")]))))
