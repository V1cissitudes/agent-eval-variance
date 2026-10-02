#!/usr/bin/env Rscript
# Variance components (docs/experiments/analysis_plan.md section 4) and D-study (section 5).
#
# Input : runs/_derived/runs_flat.csv, written by 01_run_summary.py (one row per run).
# Output: results/02_variance_components.csv  per experiment x condition x outcome
#         results/02_fixed_effects.csv        pooled LMM (primary) and logit GLMM (sensitivity)
#         results/02_dstudy.csv               SE, power and ranking-flip probability by (n_q, n_r)
#         figures/02_<experiment>_dstudy.png
#
# Design per condition: question q / seed s (nested in q) / same-seed replicate r.
#   y = mu + u_q + w_qs + e_qsr ; var(e) = serving-side nondeterminism, var(w) = sampling.
# With one replicate per seed, var(w) and var(e) cannot be separated and are reported together.

suppressPackageStartupMessages({
  library(lme4)
  library(glmmTMB)
})

INPUT <- "runs/_derived/runs_flat.csv"
DELTA <- 0.03   # smallest difference worth detecting (decisions.md, 2026-10-03)
ALPHA <- 0.05
N_Q <- c(25, 50, 100, 200, 300, 500, 1000)
N_R <- c(1, 2, 3, 5, 10, 20)

if (!file.exists(INPUT)) {
  message("02_variance_components: ", INPUT, " not found, skipping")
  quit(status = 0)
}
runs <- read.csv(INPUT, stringsAsFactors = FALSE)
if (nrow(runs) == 0) {
  message("02_variance_components: no runs, skipping")
  quit(status = 0)
}
runs$cond <- sprintf("T=%s|cache=%s|think=%s", runs$temperature, runs$prefix_cache, runs$thinking)
runs$q <- factor(runs$question_id)
runs$qs <- factor(paste(runs$question_id, runs$seed_index))

# Method-of-moments (nested ANOVA) estimates; used when REML cannot fit degenerate data, e.g.
# temperature 0 where runs of one question are identical and the residual variance is exactly 0.
pooled_within <- function(y, g) {
  v <- tapply(y, g, var)
  n <- tapply(y, g, length)
  ok <- n > 1
  if (!any(ok)) return(0)
  sum(v[ok] * (n[ok] - 1)) / sum(n[ok] - 1)
}
moments <- function(y, q, qs, has_rep) {
  cell_means <- tapply(y, qs, mean)
  cell_q <- tapply(as.character(q), qs, function(x) x[1])
  q_means <- tapply(y, q, mean)
  n_per_q <- mean(tapply(y, q, length))
  if (has_rep) {
    r <- mean(tapply(y, qs, length))
    s <- mean(tapply(names(cell_means), cell_q, length))
    v_rep <- pooled_within(y, qs)
    v_seed <- max(0, pooled_within(cell_means, cell_q) - v_rep / r)
    v_q <- max(0, var(q_means) - v_seed / s - v_rep / (s * r))
    return(c(q = v_q, seed = v_seed, rep = v_rep))
  }
  v_run <- pooled_within(y, q)
  c(q = max(0, var(q_means) - v_run / n_per_q), seed = NA, rep = NA, run = v_run)
}

# Fit y ~ 1 + (1|q) [+ (1|q:s)] by REML; a constant response means every component is zero.
condition_components <- function(d, outcome) {
  y <- d[[outcome]]
  has_rep <- max(d$replicate) > 0
  out <- data.frame(
    n_questions = length(unique(d$q)), n_seeds = length(unique(d$seed_index)),
    n_replicates = max(d$replicate) + 1, n_runs = nrow(d), mean = mean(y),
    var_question = 0, var_seed = NA_real_, var_replicate = NA_real_, var_run = 0,
    singular = FALSE, note = ""
  )
  if (var(y) == 0) {
    out$var_seed <- if (has_rep) 0 else NA
    out$var_replicate <- if (has_rep) 0 else NA
    out$note <- "constant response"
    return(out)
  }
  form <- if (has_rep) y ~ 1 + (1 | q) + (1 | qs) else y ~ 1 + (1 | q)
  d$y <- y
  fit <- tryCatch(suppressMessages(suppressWarnings(lmer(form, data = d, REML = TRUE))),
                  error = function(e) e)
  if (inherits(fit, "error")) {
    m <- moments(y, d$q, d$qs, has_rep)
    out$var_question <- m[["q"]]
    if (has_rep) {
      out$var_seed <- m[["seed"]]
      out$var_replicate <- m[["rep"]]
      out$var_run <- m[["seed"]] + m[["rep"]]
    } else {
      out$var_run <- m[["run"]]
    }
    out$note <- paste("REML failed (", conditionMessage(fit), "); method-of-moments estimates")
    return(out)
  }
  vc <- as.data.frame(VarCorr(fit))
  get <- function(g) if (any(vc$grp == g)) vc$vcov[vc$grp == g] else 0
  out$var_question <- get("q")
  if (has_rep) {
    out$var_seed <- get("qs")
    out$var_replicate <- get("Residual")
    out$var_run <- out$var_seed + out$var_replicate
  } else {
    out$var_run <- get("Residual")
    out$note <- "one replicate per seed: seed and serving variance not separable"
  }
  out$singular <- isSingular(fit)
  out
}

components <- list()
for (exp in unique(runs$experiment)) {
  for (cond in unique(runs$cond[runs$experiment == exp])) {
    d <- runs[runs$experiment == exp & runs$cond == cond, ]
    for (outcome in c("em", "f1")) {
      row <- condition_components(d, outcome)
      components[[length(components) + 1]] <- cbind(
        experiment = exp, condition = cond, outcome = outcome, row
      )
    }
  }
}
components <- do.call(rbind, components)
total <- components$var_question + components$var_run
components$share_question <- ifelse(total > 0, components$var_question / total, NA)
components$share_seed <- ifelse(total > 0, components$var_seed / total, NA)
components$share_replicate <- ifelse(total > 0, components$var_replicate / total, NA)
dir.create("results", showWarnings = FALSE)
write.csv(components, "results/02_variance_components.csv", row.names = FALSE)

# Pooled models across conditions: fixed effects of the factors + question x condition variance.
fixed <- list()
qc_var <- list()
for (exp in unique(runs$experiment)) {
  d <- runs[runs$experiment == exp, ]
  d$qc <- factor(paste(d$question_id, d$cond))
  d$qcs <- factor(paste(d$question_id, d$cond, d$seed_index))
  factors <- c("temperature", "prefix_cache", "thinking")
  varying <- factors[sapply(factors, function(f) length(unique(d[[f]])) > 1)]
  for (f in varying) d[[f]] <- factor(d[[f]])
  rhs_fixed <- if (length(varying)) paste(varying, collapse = " * ") else "1"
  # the seed level is identifiable only when seeds have same-seed replicates
  rhs_random <- if (max(d$replicate) > 0) "(1 | q) + (1 | qc) + (1 | qcs)" else "(1 | q) + (1 | qc)"
  lmm <- tryCatch(suppressMessages(suppressWarnings(
    lmer(as.formula(paste("em ~", rhs_fixed, "+", rhs_random)), data = d, REML = TRUE)
  )), error = function(e) e)
  if (inherits(lmm, "error")) {
    message("02_variance_components: pooled LMM failed for ", exp, ": ", conditionMessage(lmm))
    qc_var[[exp]] <- 0
    next
  }
  co <- summary(lmm)$coefficients
  fixed[[length(fixed) + 1]] <- data.frame(
    experiment = exp, model = "LMM (primary)", term = rownames(co),
    estimate = co[, "Estimate"], se = co[, "Std. Error"], row.names = NULL
  )
  vc <- as.data.frame(VarCorr(lmm))
  qc_var[[exp]] <- if (any(vc$grp == "qc")) vc$vcov[vc$grp == "qc"] else 0
  glmm <- tryCatch(
    suppressWarnings(glmmTMB(as.formula(paste("em ~", rhs_fixed, "+", rhs_random)),
                             data = d, family = binomial)),
    error = function(e) NULL
  )
  if (!is.null(glmm)) {
    cg <- summary(glmm)$coefficients$cond
    fixed[[length(fixed) + 1]] <- data.frame(
      experiment = exp, model = "logit GLMM (sensitivity)", term = rownames(cg),
      estimate = cg[, "Estimate"], se = cg[, "Std. Error"], row.names = NULL
    )
  }
}
if (length(fixed)) {
  write.csv(do.call(rbind, fixed), "results/02_fixed_effects.csv", row.names = FALSE)
}

# D-study on EM: SE of one system's accuracy, and power / ranking-flip probability for a
# DELTA difference between two conditions evaluated on the same questions.
z <- qnorm(1 - ALPHA / 2)
dstudy <- list()
em_rows <- components[components$outcome == "em", ]
for (i in seq_len(nrow(em_rows))) {
  r <- em_rows[i, ]
  vqc <- qc_var[[r$experiment]]
  grid <- expand.grid(n_q = N_Q, n_r = N_R)
  grid$se_accuracy <- sqrt(r$var_question / grid$n_q + r$var_run / (grid$n_q * grid$n_r))
  grid$se_difference <- sqrt(2 * (vqc / grid$n_q + r$var_run / (grid$n_q * grid$n_r)))
  grid$power_delta <- ifelse(grid$se_difference > 0,
                             pnorm(DELTA / grid$se_difference - z), 1)
  grid$p_rank_flip <- ifelse(grid$se_difference > 0, pnorm(-DELTA / grid$se_difference), 0)
  dstudy[[i]] <- cbind(experiment = r$experiment, condition = r$condition,
                       var_question = r$var_question, var_qc = vqc, var_run = r$var_run, grid)
}
dstudy <- do.call(rbind, dstudy)
write.csv(dstudy, "results/02_dstudy.csv", row.names = FALSE)

dir.create("figures", showWarnings = FALSE)
for (exp in unique(dstudy$experiment)) {
  dd <- dstudy[dstudy$experiment == exp, ]
  conds <- unique(dd$condition)
  png(sprintf("figures/02_%s_dstudy.png", exp), width = 520 * length(conds), height = 460, res = 110)
  par(mfrow = c(1, length(conds)), mar = c(4.2, 4.2, 3, 1))
  for (cond in conds) {
    dc <- dd[dd$condition == cond, ]
    plot(NA, xlim = range(N_Q), ylim = c(0, 1), log = "x", xlab = "questions (n_q)",
         ylab = sprintf("power to detect %.0f pp", 100 * DELTA), main = cond, cex.main = 0.8)
    abline(h = 0.8, lty = 2, col = "grey50")
    cols <- hcl.colors(length(N_R), "Viridis")
    for (k in seq_along(N_R)) {
      dk <- dc[dc$n_r == N_R[k], ]
      lines(dk$n_q, dk$power_delta, col = cols[k], lwd = 2)
    }
    legend("bottomright", legend = paste("n_r =", N_R), col = cols, lwd = 2, cex = 0.7, bty = "n")
  }
  dev.off()
}
message(sprintf("02_variance_components: %d condition x outcome rows, D-study written",
                nrow(components)))
