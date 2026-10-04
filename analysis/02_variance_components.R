#!/usr/bin/env Rscript
# Variance components (docs/experiments/analysis_plan.md section 4) and D-study (section 5).
#
# Input : runs/_derived/runs_flat.csv, written by 01_run_summary.py (one row per run).
# Output: results/02_variance_components.csv  per experiment x condition x outcome
#         results/02_fixed_effects.csv        pooled LMM (primary) and logit GLMM (sensitivity)
#         results/02_dstudy.csv               SE of one system's accuracy by (n_q, n_r independent runs)
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
if (!"arm" %in% names(runs)) runs$arm <- "default"
if (!"precision" %in% names(runs)) runs$precision <- ""
runs$cond <- sprintf("%s|%s|T=%s|cache=%s|think=%s", runs$precision, runs$arm, runs$temperature,
                     runs$prefix_cache, runs$thinking)
runs$q <- factor(runs$question_id)
runs$qs <- factor(paste(runs$question_id, runs$seed_index))

# Method-of-moments (nested ANOVA) estimates; used when REML cannot fit degenerate data, e.g.
# temperature 0 where runs of one question are identical and the residual variance is exactly 0.
pooled_within <- function(y, g) {
  v <- tapply(y, g, var)
  n <- tapply(y, g, length)
  ok <- !is.na(n) & n > 1  # tapply gives NA for factor levels absent from this subset
  if (!any(ok)) return(0)
  sum(v[ok] * (n[ok] - 1)) / sum(n[ok] - 1)
}
# Design of one condition: same-seed replicates present? more than one seed per question?
#   "seed+replicate": S >= 2 seeds x R >= 2 replicates -> question, seed and replicate components
#   "single-seed":    S = 1, R >= 2 (greedy arm) -> question and replicate; the seed level is not
#                     identifiable (and has no role at temperature 0), reported as NA
#   "no-replicate":   R = 1 -> question and run (seed and serving variance not separable)
design_of <- function(d) {
  has_rep <- max(d$replicate) > 0
  seeds_per_q <- max(tapply(d$seed_index, d$q, function(x) length(unique(x))), na.rm = TRUE)
  if (!has_rep) return("no-replicate")
  if (seeds_per_q < 2) return("single-seed")
  "seed+replicate"
}

moments <- function(y, q, qs, design) {
  q_means <- tapply(y, q, mean)
  n_per_q <- mean(tapply(y, q, length), na.rm = TRUE)
  if (design == "seed+replicate") {
    cell_means <- tapply(y, qs, mean)
    cell_q <- tapply(as.character(q), qs, function(x) x[1])
    r <- mean(tapply(y, qs, length), na.rm = TRUE)
    s <- mean(tapply(names(cell_means), cell_q, length), na.rm = TRUE)
    v_rep <- pooled_within(y, qs)
    v_seed <- max(0, pooled_within(cell_means, cell_q) - v_rep / r)
    v_q <- max(0, var(q_means, na.rm = TRUE) - v_seed / s - v_rep / (s * r))
    return(c(q = v_q, seed = v_seed, rep = v_rep, run = v_seed + v_rep))
  }
  v_within <- pooled_within(y, q)
  v_q <- max(0, var(q_means, na.rm = TRUE) - v_within / n_per_q)
  if (design == "single-seed") return(c(q = v_q, seed = NA, rep = v_within, run = v_within))
  c(q = v_q, seed = NA, rep = NA, run = v_within)
}

# Question-level (cluster) bootstrap of the method-of-moments estimates -> 95% intervals.
# Resampling whole questions keeps the seed / replicate structure inside each question intact.
boot_components <- function(d, outcome, design, B = 500, seed = 20261003) {
  set.seed(seed)
  idx_by_q <- split(seq_len(nrow(d)), as.character(d$q))
  draws <- replicate(B, {
    pick <- sample(names(idx_by_q), length(idx_by_q), replace = TRUE)
    rows <- unlist(idx_by_q[pick], use.names = FALSE)
    newq <- rep(seq_along(pick), times = lengths(idx_by_q[pick]))
    m <- moments(d[[outcome]][rows], factor(newq), factor(paste(newq, d$seed_index[rows])), design)
    c(question = m[["q"]], seed = m[["seed"]], replicate = m[["rep"]], run = m[["run"]])
  })
  q <- apply(draws, 1, function(x) if (all(is.na(x))) c(NA, NA) else
    quantile(x, probs = c(0.025, 0.975), na.rm = TRUE))
  setNames(as.vector(q), paste0("var_", rep(rownames(draws), each = 2), c("_lo", "_hi")))
}

# Point estimates by REML where identifiable and well-posed, otherwise method of moments; the
# method-of-moments point estimates are always reported too, because the bootstrap intervals are
# computed with that estimator.
condition_components <- function(d, outcome) {
  y <- d[[outcome]]
  design <- design_of(d)
  m <- moments(y, d$q, d$qs, design)
  out <- data.frame(
    design = design, n_questions = length(unique(d$q)), n_seeds = length(unique(d$seed_index)),
    n_replicates = max(d$replicate) + 1, n_runs = nrow(d), mean = mean(y),
    var_question = m[["q"]], var_seed = m[["seed"]], var_replicate = m[["rep"]], var_run = m[["run"]],
    point_estimator = "method of moments", singular = NA, note = "",
    var_question_mom = m[["q"]], var_seed_mom = m[["seed"]], var_replicate_mom = m[["rep"]],
    var_run_mom = m[["run"]]
  )
  if (var(y) == 0) {
    out$note <- "constant response"
    return(out)
  }
  innermost <- if (design == "seed+replicate") d$qs else d$q
  if (pooled_within(y, innermost) == 0) {
    # residual variance exactly 0: lmer errors or silently mis-fits, so keep the moments
    out$note <- "no within-group variation (residual variance is 0)"
    return(out)
  }
  d$y <- y
  form <- if (design == "seed+replicate") y ~ 1 + (1 | q) + (1 | qs) else y ~ 1 + (1 | q)
  fit <- tryCatch(suppressMessages(suppressWarnings(lmer(form, data = d, REML = TRUE))),
                  error = function(e) e)
  if (inherits(fit, "error")) {
    out$note <- paste("REML failed:", conditionMessage(fit))
    return(out)
  }
  vc <- as.data.frame(VarCorr(fit))
  get <- function(g) if (any(vc$grp == g)) vc$vcov[vc$grp == g] else 0
  out$var_question <- get("q")
  if (design == "seed+replicate") {
    out$var_seed <- get("qs")
    out$var_replicate <- get("Residual")
    out$var_run <- out$var_seed + out$var_replicate
  } else if (design == "single-seed") {
    out$var_replicate <- get("Residual")
    out$var_run <- out$var_replicate
    out$note <- "one seed per question: seed level not identifiable (NA)"
  } else {
    out$var_run <- get("Residual")
    out$note <- "one replicate per seed: seed and serving variance not separable"
  }
  out$point_estimator <- "REML"
  out$singular <- isSingular(fit)
  out
}

components <- list()
for (exp in unique(runs$experiment)) {
  for (cond in unique(runs$cond[runs$experiment == exp])) {
    d <- droplevels(runs[runs$experiment == exp & runs$cond == cond, ])
    for (outcome in c("em", "f1")) {
      row <- condition_components(d, outcome)
      ci <- as.data.frame(t(boot_components(d, outcome, row$design)))
      ci$ci_estimator <- "method of moments, question-level bootstrap (500)"
      components[[length(components) + 1]] <- cbind(
        experiment = exp, condition = cond, outcome = outcome, row, ci
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
for (exp in unique(runs$experiment)) {
  d <- droplevels(runs[runs$experiment == exp, ])
  d$qc <- factor(paste(d$question_id, d$cond))
  d$qcs <- factor(paste(d$question_id, d$cond, d$seed_index))
  factors <- c("precision", "arm", "temperature", "prefix_cache", "thinking")
  varying <- factors[sapply(factors, function(f) length(unique(d[[f]])) > 1)]
  # drop a factor that is fully determined by another one (e.g. temperature by arm "greedy"),
  # otherwise the fixed-effect design is rank deficient and the GLMM returns NaN
  determined <- function(f, g) all(tapply(d[[f]], d[[g]], function(x) length(unique(x))) == 1)
  for (f in rev(varying)) {
    others <- setdiff(varying, f)
    if (any(sapply(others, function(g) determined(f, g)))) varying <- others
  }
  for (f in varying) d[[f]] <- factor(d[[f]])
  rhs_fixed <- if (length(varying)) paste(varying, collapse = " * ") else "1"
  # the seed level is identifiable only when seeds have same-seed replicates
  rhs_random <- if (max(d$replicate) > 0) "(1 | q) + (1 | qc) + (1 | qcs)" else "(1 | q) + (1 | qc)"
  lmm <- tryCatch(suppressMessages(suppressWarnings(
    lmer(as.formula(paste("em ~", rhs_fixed, "+", rhs_random)), data = d, REML = TRUE)
  )), error = function(e) e)
  if (inherits(lmm, "error")) {
    message("02_variance_components: pooled LMM failed for ", exp, ": ", conditionMessage(lmm))
    next
  }
  co <- summary(lmm)$coefficients
  fixed[[length(fixed) + 1]] <- data.frame(
    experiment = exp, model = "LMM (primary)", term = rownames(co),
    estimate = co[, "Estimate"], se = co[, "Std. Error"], row.names = NULL
  )
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

# D-study on EM for a single system: SE of the accuracy with n_q questions and n_r independent runs
# per question (a fresh seed for every run). A fresh-seed run carries var_seed + var_replicate, so
# var_run applies directly. Comparisons between conditions (paired questions and seeds) are in 04.
dstudy <- list()
em_rows <- components[components$outcome == "em", ]
for (i in seq_len(nrow(em_rows))) {
  r <- em_rows[i, ]
  grid <- expand.grid(n_q = N_Q, n_r = N_R)
  grid$se_accuracy <- sqrt(r$var_question / grid$n_q + r$var_run / (grid$n_q * grid$n_r))
  dstudy[[i]] <- cbind(experiment = r$experiment, condition = r$condition,
                       var_question = r$var_question, var_run = r$var_run, grid)
}
dstudy <- do.call(rbind, dstudy)
write.csv(dstudy, "results/02_dstudy.csv", row.names = FALSE)

dir.create("figures", showWarnings = FALSE)
for (exp in unique(dstudy$experiment)) {
  dd <- dstudy[dstudy$experiment == exp, ]
  conds <- unique(dd$condition)
  # type = "cairo": no EXIF/ICC chunks (the macOS quartz default adds them)
  png(sprintf("figures/02_%s_dstudy.png", exp), width = 520 * length(conds), height = 460, res = 110,
      type = "cairo")
  par(mfrow = c(1, length(conds)), mar = c(4.2, 4.2, 3, 1))
  for (cond in conds) {
    dc <- dd[dd$condition == cond, ]
    plot(NA, xlim = range(N_Q), ylim = c(0, max(dd$se_accuracy)), log = "x",
         xlab = "questions (n_q)", ylab = "SE of accuracy", main = cond, cex.main = 0.8)
    abline(h = 0.01, lty = 2, col = "grey50")
    cols <- hcl.colors(length(N_R), "Viridis")
    for (k in seq_along(N_R)) {
      dk <- dc[dc$n_r == N_R[k], ]
      lines(dk$n_q, dk$se_accuracy, col = cols[k], lwd = 2)
    }
    legend("topright", legend = paste("n_r =", N_R), col = cols, lwd = 2, cex = 0.7, bty = "n")
  }
  dev.off()
}
message(sprintf("02_variance_components: %d condition x outcome rows, D-study written",
                nrow(components)))
