#!/usr/bin/env Rscript
# H3 and H4 (docs/experiments/E01_design.md), precision ladder BF16 / FP8 / INT4 (decided 2026-10-03).
# BF16 rows come from E01a (arms greedy and qwen), FP8 / INT4 rows from E01b; all use seed base
# 20261003 on the same 300 questions, so precisions are paired by question and seed.
#
# H3: with prefix caching on, same-seed divergence increases as precision decreases
#     -> Cochran-Armitage trend test (prop.trend.test, two-sided, scores = weight bits 16/8/4, i.e.
#        unequally spaced) with the direction of the trend reported separately; per arm,
#        for all pairs and for the first (cache miss vs hit) pair separately; cache-off counts with
#        exact upper bounds (a precision effect there would contradict 2609.04748).
# H4: precision changes accuracy -> paired difference to BF16 on per-question mean EM,
#     two-sided test (Holm across all comparisons) and TOST equivalence with margin +-3 pp
#     (equivalent when the 90% CI lies inside the margin).
#
# Input : runs/_derived/runs_flat.csv
#     Sensitivity for H3: the seed groups of one question are not independent, so the divergence
#     difference to BF16 also gets a question-level bootstrap interval (same questions for both).
#     A logistic precision x cache interaction is not fitted: with caching off divergence is 0 in every
#     cell (H1), so the model is not estimable; the cache-off counts and upper bounds are reported.
#     Also: LMM fixed effect of precision on EM on the pooled BF16 / FP8 / INT4 data.
# Completeness: only (question, seed, replicate) cells present for every precision are compared, the
# Holm family is the full planned set (2 quantized levels x 2 arms x 2 cache settings), and missing
# comparisons are reported as NA instead of shrinking the family.
#
# Output: results/06_h3_divergence_trend.csv, results/06_h3_divergence_bootstrap.csv,
#         results/06_h4_accuracy_equivalence.csv, results/06_h4_lmm_precision.csv

INPUT <- "runs/_derived/runs_flat.csv"
MARGIN <- 0.03
ARMS <- c("greedy", "qwen")

if (!file.exists(INPUT)) {
  message("06_h3_h4_precision: ", INPUT, " not found, skipping")
  quit(status = 0)
}
runs <- read.csv(INPUT, stringsAsFactors = FALSE)
for (f in c("arm", "precision")) if (!f %in% names(runs)) runs[[f]] <- ""
d <- runs[(runs$experiment == "E01a" & runs$arm %in% ARMS) | runs$experiment == "E01b", ]
d <- d[d$arm %in% ARMS, ]

bits <- function(p) {
  p <- tolower(p)
  ifelse(grepl("bf16|fp16|f16", p), 16, ifelse(grepl("fp8|int8|w8", p), 8,
         ifelse(grepl("int4|awq|w4|q4", p), 4, NA)))
}
d$bits <- bits(d$precision)
if (length(unique(na.omit(d$bits))) < 2) {
  message("06_h3_h4_precision: fewer than two precisions available, skipping")
  quit(status = 0)
}

# ---------------- completeness ----------------
d$q <- d$question_id
d$cell <- paste(d$arm, d$prefix_cache, d$question_id, d$seed_index, d$replicate)
d$group <- paste(d$arm, d$prefix_cache, d$question_id, d$seed_index)
bf16_cells <- unique(d$cell[d$bits == 16])
quantized <- sort(unique(d$precision[d$bits < 16]))
coverage <- data.frame(
  precision = c("bf16", quantized),
  cells = sapply(c("bf16", quantized), function(p) sum(d$precision == p)),
  matched_to_bf16 = sapply(c("bf16", quantized), function(p) sum(d$cell[d$precision == p] %in% bf16_cells))
)
print(coverage, row.names = FALSE)

# same-seed groups: one row per (precision, arm, cache, question, seed) with exactly R replicates
grp <- aggregate(cbind(n = em) ~ precision + bits + arm + prefix_cache + q + seed_index + group,
                 data = d, FUN = length)
differs <- tapply(seq_len(nrow(d)), paste(d$precision, d$group), function(i) {
  length(unique(d$answer[i])) > 1 || length(unique(d$actions[i])) > 1
})
grp$differs <- as.vector(differs[paste(grp$precision, grp$group)])
grp <- grp[grp$n >= 2, ]

# ---------------- H3: divergence trend + question-level bootstrap ----------------
set.seed(20261003)
h3 <- list()
h3_boot <- list()
for (arm in ARMS) for (cache in c("on", "off")) for (position in c("all", "first", "later")) {
  g <- grp[grp$arm == arm & grp$prefix_cache == cache, ]
  if (position == "first") g <- g[g$seed_index == 0, ]
  if (position == "later") g <- g[g$seed_index > 0, ]
  if (!nrow(g)) next
  # trend uses groups present for every available precision (paired across precisions)
  precs <- unique(g$precision)
  common <- Reduce(intersect, lapply(precs, function(p) g$group[g$precision == p]))
  gc <- g[g$group %in% common, ]
  tab <- aggregate(differs ~ precision + bits, data = gc, FUN = function(x) c(k = sum(x), n = length(x)))
  tab <- data.frame(precision = tab$precision, bits = tab$bits, k = tab$differs[, "k"],
                    n = tab$differs[, "n"])
  tab <- tab[order(-tab$bits), ]
  rate <- tab$k / tab$n
  p_trend <- if (nrow(tab) >= 2 && sum(tab$k) > 0 && sum(tab$k) < sum(tab$n)) {
    suppressWarnings(prop.trend.test(tab$k, tab$n, score = tab$bits)$p.value)
  } else NA
  direction <- if (nrow(tab) < 2 || diff(range(rate)) == 0) "none" else
    if (rate[which.min(tab$bits)] > rate[which.max(tab$bits)]) "more divergence at lower precision" else
      "less divergence at lower precision"
  for (i in seq_len(nrow(tab))) {
    ci <- binom.test(tab$k[i], tab$n[i])$conf.int
    h3[[length(h3) + 1]] <- data.frame(
      arm = arm, prefix_cache = cache, pair_position = position, precision = tab$precision[i],
      bits = tab$bits[i], seed_groups = tab$n[i], differing = tab$k[i], rate = rate[i],
      ci_low = ci[1], ci_high = ci[2], p_trend_two_sided = p_trend, trend_direction = direction,
      note = "CP interval assumes independent seed groups; see bootstrap file for clustering"
    )
  }
  # question-level bootstrap of the divergence difference to BF16 (same questions for both)
  for (p in setdiff(precs, "bf16")) {
    gb <- gc[gc$precision == "bf16", ]
    gp <- gc[gc$precision == p, ]
    qs <- unique(gb$q)
    rate_diff <- function(pick) {
      rb <- unlist(lapply(pick, function(x) gb$differs[gb$q == x]))
      rp <- unlist(lapply(pick, function(x) gp$differs[gp$q == x]))
      mean(rp) - mean(rb)
    }
    ib <- split(gb$differs, gb$q)
    ip <- split(gp$differs, gp$q)
    boot <- replicate(1000, {
      pick <- sample(qs, length(qs), replace = TRUE)
      mean(unlist(ip[pick])) - mean(unlist(ib[pick]))
    })
    h3_boot[[length(h3_boot) + 1]] <- data.frame(
      arm = arm, prefix_cache = cache, pair_position = position, precision = p,
      rate_bf16 = mean(gb$differs), rate_quantized = mean(gp$differs),
      rate_diff = mean(gp$differs) - mean(gb$differs),
      ci_low = quantile(boot, 0.025), ci_high = quantile(boot, 0.975), questions = length(qs),
      row.names = NULL
    )
  }
}
dir.create("results", showWarnings = FALSE)
write.csv(do.call(rbind, h3), "results/06_h3_divergence_trend.csv", row.names = FALSE)
if (length(h3_boot)) {
  write.csv(do.call(rbind, h3_boot), "results/06_h3_divergence_bootstrap.csv", row.names = FALSE)
}

# ---------------- H4: accuracy vs BF16, paired, with TOST ----------------
PLANNED_BITS <- c(8, 4)  # FP8, INT4 (decisions.md 2026-10-03)
h4 <- list()
for (arm in ARMS) for (cache in c("off", "on")) for (b in PLANNED_BITS) {
  prec <- unique(d$precision[d$bits == b])
  row <- data.frame(arm = arm, prefix_cache = cache, bits = b,
                    precision = if (length(prec)) prec[1] else NA, n_questions = NA,
                    matched_cells = NA, acc_bf16 = NA, acc_quantized = NA, diff = NA,
                    ci95_low = NA, ci95_high = NA, ci90_low = NA, ci90_high = NA,
                    p_two_sided = NA, p_tost = NA, note = "")
  if (!length(prec)) {
    row$note <- "precision not available yet"
    h4[[length(h4) + 1]] <- row
    next
  }
  base <- d[d$bits == 16 & d$arm == arm & d$prefix_cache == cache, ]
  other <- d[d$precision == prec[1] & d$arm == arm & d$prefix_cache == cache, ]
  cells <- intersect(base$cell, other$cell)
  base <- base[base$cell %in% cells, ]
  other <- other[other$cell %in% cells, ]
  if (length(cells) == 0) {
    row$note <- "no matched cells"
    h4[[length(h4) + 1]] <- row
    next
  }
  mb <- tapply(base$em, base$q, mean)
  mo <- tapply(other$em, other$q, mean)[names(mb)]
  diff <- mo - mb
  n <- length(diff)
  est <- mean(diff)
  se <- sd(diff) / sqrt(n)
  row$n_questions <- n
  row$matched_cells <- length(cells)
  row$acc_bf16 <- mean(mb)
  row$acc_quantized <- mean(mo)
  row$diff <- est
  if (!is.na(se) && se > 0) {
    row$ci95_low <- est - qt(0.975, n - 1) * se
    row$ci95_high <- est + qt(0.975, n - 1) * se
    row$ci90_low <- est - qt(0.95, n - 1) * se
    row$ci90_high <- est + qt(0.95, n - 1) * se
    row$p_two_sided <- 2 * pt(-abs(est / se), df = n - 1)
    p_lower <- pt((est + MARGIN) / se, df = n - 1, lower.tail = FALSE)
    p_upper <- pt((est - MARGIN) / se, df = n - 1)
    row$p_tost <- max(p_lower, p_upper)
  } else {
    # every per-question difference identical: the test statistics are undefined; decide from the
    # constant difference itself and say so
    row$ci95_low <- row$ci95_high <- row$ci90_low <- row$ci90_high <- est
    row$p_two_sided <- if (est == 0) 1 else 0
    row$p_tost <- if (abs(est) < MARGIN) 0 else 1
    row$note <- "degenerate: constant per-question difference (no variance)"
  }
  if (length(cells) < length(unique(d$cell[d$bits == 16 & d$arm == arm & d$prefix_cache == cache]))) {
    row$note <- trimws(paste(row$note, "incomplete: fewer matched cells than BF16"))
  }
  h4[[length(h4) + 1]] <- row
}
h4 <- do.call(rbind, h4)
family <- nrow(h4)  # full planned family, also when some comparisons are still missing
h4$p_two_sided_holm <- p.adjust(h4$p_two_sided, method = "holm", n = family)
h4$p_tost_holm <- p.adjust(h4$p_tost, method = "holm", n = family)
h4$equivalent_within_3pp <- h4$p_tost_holm < 0.05  # pre-registered: TOST with Holm correction
write.csv(h4, "results/06_h4_accuracy_equivalence.csv", row.names = FALSE)

# ---------------- H4 supplement: LMM fixed effect of precision (pooled) ----------------
suppressPackageStartupMessages(library(lme4))
d$precision_f <- relevel(factor(d$precision), ref = unique(d$precision[d$bits == 16])[1])
d$qc <- factor(paste(d$q, d$precision, d$arm, d$prefix_cache))
d$qcs <- factor(paste(d$qc, d$seed_index))
lmm <- tryCatch(suppressMessages(suppressWarnings(
  lmer(em ~ precision_f * arm * prefix_cache + (1 | q) + (1 | qc) + (1 | qcs), data = d)
)), error = function(e) e)
if (!inherits(lmm, "error")) {
  co <- summary(lmm)$coefficients
  write.csv(data.frame(term = rownames(co), estimate = co[, "Estimate"], se = co[, "Std. Error"],
                       row.names = NULL), "results/06_h4_lmm_precision.csv", row.names = FALSE)
}
message(sprintf("06_h3_h4_precision: H3 %d rows, H4 %d planned comparisons (%d available)",
                length(h3), nrow(h4), sum(!is.na(h4$diff))))
