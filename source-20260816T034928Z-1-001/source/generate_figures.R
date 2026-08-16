suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(survival)
  library(survminer)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
root <- normalizePath(file.path(dirname(file_arg), ".."))
fig_dir <- file.path(root, "results", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

selected_root <- file.path(root, "results", "sensitivity", "dropout040_adam")
performance <- read.csv(file.path(selected_root, "metrics", "performance_summary.csv"), check.names = FALSE)
fold_metrics <- read.csv(file.path(selected_root, "metrics", "fold_metrics.csv"), check.names = FALSE)
predictions <- read.csv(file.path(selected_root, "predictions", "all_predictions.csv"), check.names = FALSE)
dataset <- read.csv(file.path(root, "data", "processed", "dataset_summary.csv"), check.names = FALSE)
validation_order <- c(
  "Internal combined 5-fold CV",
  "External CGGA to TCGA",
  "External TCGA to CGGA"
)
performance$validation <- factor(performance$validation, levels = validation_order)

model_labels <- c(
  cox_residual_kg_attention_nohazard = "CoxRes-KGA",
  xgboost_aft = "XGBoost AFT",
  random_survival_forest = "Random survival forest",
  survival_svm = "Survival SVM",
  linear_regression = "Linear regression",
  regular_neural_network = "Regular neural network"
)
performance$model_label <- unname(model_labels[performance$model])
fold_metrics$model_label <- unname(model_labels[fold_metrics$model])
predictions$model_label <- unname(model_labels[predictions$model])
comparison_order <- c(
  "CoxRes-KGA", "Random survival forest", "Survival SVM",
  "XGBoost AFT", "Regular neural network", "Linear regression"
)
performance$model_label <- factor(performance$model_label, levels = rev(comparison_order))

theme_report <- function() {
  theme_minimal(base_family = "Arial", base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 13, color = "#17365D"),
      plot.subtitle = element_text(size = 9, color = "#4D4D4D"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      axis.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold", color = "#17365D")
    )
}

save_plot <- function(name, plot, width = 8.2, height = 5.2) {
  ggsave(file.path(fig_dir, name), plot, width = width, height = height, dpi = 300, bg = "white")
}

# Figure 1: cohort composition.
composition <- dataset %>%
  select(dataset, events, censored) %>%
  pivot_longer(c(events, censored), names_to = "status", values_to = "n")
p1 <- ggplot(composition, aes(dataset, n, fill = status)) +
  geom_col(width = 0.68) +
  geom_text(aes(label = n), position = position_stack(vjust = 0.5), color = "white", fontface = "bold") +
  scale_fill_manual(
    values = c(events = "#B4463A", censored = "#4C78A8"),
    labels = c(events = "Death observed", censored = "Censored")
  ) +
  labs(title = "Analysis cohorts and observed outcomes", x = NULL, y = "Patients", fill = NULL) +
  theme_report()
save_plot("figure_01_cohort_composition.png", p1, 8.2, 4.8)

# Figure 2: publication-style CoxRes-KGA architecture diagram.
source(file.path(root, "source", "draw_architecture.R"))
draw_coxres_kga(file.path(fig_dir, "figure_02_model_architecture.png"))

# Figure 3: C-index comparison.
p3 <- ggplot(
  performance,
  aes(c_index, model_label)
) +
  geom_vline(xintercept = 0.5, linetype = 2, color = "#999999") +
  geom_segment(aes(x = 0.5, xend = c_index, yend = model_label), color = "#CFD8DF") +
  geom_point(size = 2.5, color = "#557A95") +
  facet_wrap(~validation, ncol = 1) +
  coord_cartesian(xlim = c(0.48, 0.83)) +
  labs(
    title = "C-index across internal and external validations",
    subtitle = "Comparable-pair concordance; higher values indicate better discrimination",
    x = "C-index", y = NULL
  ) +
  theme_report()
save_plot("figure_03_cindex_comparison.png", p3, 9.2, 8.5)

# Figure 4: mean AUC and reported IBS.
metric_long <- performance %>%
  select(validation, model, model_label, auc_mean, ibs) %>%
  pivot_longer(c(auc_mean, ibs), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(
    metric,
    auc_mean = "Mean time-dependent AUC (higher is better)",
    ibs = "Reported IBS (lower is better)"
  ))
p4 <- ggplot(
  metric_long,
  aes(value, model_label)
) +
  geom_col(width = 0.7, fill = "#6F91A6") +
  facet_grid(metric ~ validation, scales = "free_x") +
  labs(title = "Discrimination and prediction error", x = NULL, y = NULL) +
  theme_report()
save_plot("figure_04_auc_ibs_comparison.png", p4, 11.5, 8.2)

# Table-linked comparison figures. These deliberately use the same visual grammar
# and model ordering so readers can move directly between Tables 5-7 and plots.
metric_titles <- c(
  c_index = "C-index\nHigher is better",
  auc_mean = "Mean time-dependent AUC\nHigher is better",
  ibs = "Reported IBS\nLower is better"
)

table_comparison_plot <- function(frame, title, subtitle, show_sd = FALSE) {
  model_order <- frame %>%
    filter(metric == "c_index") %>%
    arrange(desc(estimate)) %>%
    pull(model_label) %>%
    as.character()

  plot_frame <- frame %>%
    mutate(
      model_label = factor(model_label, levels = rev(model_order)),
      metric_label = factor(unname(metric_titles[metric]), levels = unname(metric_titles)),
      low = if (show_sd) estimate - sd else estimate,
      high = if (show_sd) estimate + sd else estimate,
      value_label = if (show_sd) {
        sprintf("%.3f (%.3f)", estimate, sd)
      } else {
        sprintf("%.3f", estimate)
      }
    )

  ggplot(plot_frame, aes(estimate, model_label)) +
    geom_segment(
      aes(x = low, xend = high, yend = model_label),
      linewidth = if (show_sd) 1.0 else 0.45,
      color = if (show_sd) "#91A4B5" else "#D5DDE4",
      lineend = "round"
    ) +
    geom_point(shape = 21, size = 3.2, stroke = 0.9, fill = "white", color = "#244B67") +
    geom_text(aes(label = value_label), hjust = -0.15, size = 3.0, color = "#17212B") +
    facet_wrap(~metric_label, scales = "free_x", nrow = 1) +
    scale_y_discrete(
      drop = FALSE,
      labels = function(x) ifelse(x == "Random survival forest", "Random survival\nforest", x)
    ) +
    scale_x_continuous(expand = expansion(mult = c(0.08, 0.34))) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
    theme_report() +
    theme(
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "#EEF3F7", color = NA),
      strip.text = element_text(face = "bold", color = "#17365D", size = 9.5, lineheight = 0.95),
      axis.text.y = element_text(color = "#263746", size = 8.8),
      plot.margin = margin(8, 24, 8, 8)
    )
}

internal_table_plot <- fold_metrics %>%
  filter(strategy == "combined_cv") %>%
  group_by(model, model_label) %>%
  summarise(
    c_index_sd = sd(c_index), auc_mean_sd = sd(auc_mean), ibs_sd = sd(ibs),
    c_index = mean(c_index), auc_mean = mean(auc_mean), ibs = mean(ibs),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(c_index, auc_mean, ibs),
    names_to = "metric", values_to = "estimate"
  ) %>%
  mutate(sd = case_when(
    metric == "c_index" ~ c_index_sd,
    metric == "auc_mean" ~ auc_mean_sd,
    TRUE ~ ibs_sd
  ))

p_table5 <- table_comparison_plot(
  internal_table_plot,
  "Internal five-fold cross-validation performance",
  "Points are fold means; horizontal intervals show ±1 SD across the five held-out folds",
  show_sd = TRUE
)
save_plot("figure_table5_internal_comparison.png", p_table5, 11.5, 5.1)

external_plot_data <- performance %>%
  filter(validation != "Internal combined 5-fold CV") %>%
  select(validation, model, model_label, c_index, auc_mean, ibs) %>%
  pivot_longer(c(c_index, auc_mean, ibs), names_to = "metric", values_to = "estimate") %>%
  mutate(sd = NA_real_)

p_table6 <- table_comparison_plot(
  external_plot_data %>% filter(validation == "External CGGA to TCGA"),
  "External validation: CGGA training to TCGA testing",
  "All models were trained in CGGA and evaluated once in the independent TCGA cohort"
)
save_plot("figure_table6_cgga_to_tcga_comparison.png", p_table6, 11.5, 5.1)

p_table7 <- table_comparison_plot(
  external_plot_data %>% filter(validation == "External TCGA to CGGA"),
  "External validation: TCGA training to CGGA testing",
  "All models were trained in TCGA and evaluated once in the independent CGGA cohort"
)
save_plot("figure_table7_tcga_to_cgga_comparison.png", p_table7, 11.5, 5.1)

# Figure 5: horizon-specific AUC across all models and validation settings.
auc_by_horizon <- performance %>%
  select(validation, model, model_label, auc_12, auc_24, auc_36, auc_60) %>%
  pivot_longer(starts_with("auc_"), names_to = "horizon", values_to = "auc") %>%
  mutate(
    horizon = as.numeric(sub("auc_", "", horizon)),
    validation = factor(
      validation,
      levels = c(
        "Internal combined 5-fold CV",
        "External CGGA to TCGA",
        "External TCGA to CGGA"
      )
    ),
    model_label = factor(model_label, levels = unname(model_labels))
  )

model_colors <- c(
  "CoxRes-KGA" = "#C7352D",
  "XGBoost AFT" = "#6B6B6B",
  "Random survival forest" = "#0072B2",
  "Survival SVM" = "#009E73",
  "Linear regression" = "#E69F00",
  "Regular neural network" = "#CC79A7"
)

# Figure 5: held-out fold estimates and fold means for every model.
fold_model_order <- c(
  "Random survival forest",
  "Survival SVM",
  "CoxRes-KGA",
  "Regular neural network",
  "Linear regression",
  "XGBoost AFT"
)
fold_long <- fold_metrics %>%
  filter(strategy == "combined_cv") %>%
  select(fold, model_label, c_index, auc_mean, ibs) %>%
  pivot_longer(c(c_index, auc_mean, ibs), names_to = "metric", values_to = "value") %>%
  mutate(
    fold = factor(fold, levels = as.character(1:5)),
    model_label = factor(model_label, levels = rev(fold_model_order)),
    metric = factor(
      metric,
      levels = c("c_index", "auc_mean", "ibs"),
      labels = c("C-index", "Mean AUC", "Reported IBS")
    )
  )
fold_means <- fold_long %>%
  group_by(model_label, metric) %>%
  summarise(value = mean(value), .groups = "drop")

p_fold <- ggplot(fold_long, aes(value, model_label, color = fold)) +
  geom_point(position = position_dodge(width = 0.52), size = 2.25, alpha = 0.9) +
  geom_point(
    data = fold_means,
    aes(value, model_label),
    inherit.aes = FALSE,
    shape = 23, size = 3.0, stroke = 0.8,
    fill = "white", color = "#1F2933"
  ) +
  facet_wrap(~metric, nrow = 1, scales = "free_x") +
  scale_color_brewer(palette = "Dark2", name = "Held-out fold") +
  scale_x_continuous(labels = function(x) sprintf("%.2f", x)) +
  labs(
    title = "Five-fold internal-validation stability by model",
    subtitle = "Colored points are held-out fold estimates; diamonds are fold means",
    x = NULL, y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    axis.text.y = element_text(size = 8.2, color = "#333333"),
    axis.text.x = element_text(size = 7.7),
    panel.spacing = unit(0.8, "lines"),
    legend.position = "bottom"
  )
save_plot("figure_fold_stability.png", p_fold, 11.2, 5.6)

p5 <- ggplot(
  auc_by_horizon,
  aes(
    horizon, auc,
    color = model_label,
    group = model_label
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.2) +
  facet_wrap(~validation, ncol = 1) +
  scale_color_manual(values = model_colors, drop = FALSE) +
  scale_x_continuous(breaks = c(12, 24, 36, 60)) +
  coord_cartesian(ylim = c(0.60, 0.95)) +
  labs(
    title = "Time-dependent AUC across models and validation settings",
    subtitle = "Higher values indicate better discrimination",
    x = "Prediction horizon (months)", y = "AUC", color = NULL
  ) +
  theme_report() +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  theme(
    legend.text = element_text(size = 8),
    panel.spacing = unit(0.7, "lines")
  )
save_plot("figure_05_time_auc.png", p5, 9.4, 9.2)

# Figure 6: internal out-of-fold Kaplan-Meier risk groups.
internal <- predictions %>%
  filter(strategy == "combined_cv", model == "cox_residual_kg_attention_nohazard") %>%
  group_by(dataset) %>%
  mutate(risk_group = ifelse(risk_score >= median(risk_score), "High risk", "Low risk")) %>%
  ungroup()
km_plots <- list()
for (cohort in c("CGGA", "TCGA")) {
  item <- internal %>% filter(dataset == cohort)
  fit <- survfit(Surv(time, event) ~ risk_group, data = item)
  rendered <- ggsurvplot(
    fit, data = item, risk.table = FALSE, pval = TRUE, conf.int = FALSE,
    xlim = c(0, 120), break.time.by = 24,
    palette = c("#B4463A", "#4C78A8"), legend.title = "",
    legend.labs = c("High risk", "Low risk"),
    title = paste("Combined-CV out-of-fold risk groups:", cohort),
    xlab = "Months", ylab = "Survival probability",
    ggtheme = theme_report()
  )
  km_plots[[cohort]] <- rendered$plot
}
save_plot("figure_06_km_internal.png", km_plots[["CGGA"]] + km_plots[["TCGA"]], 11.0, 5.2)

# Figure 7: external calibration of CoxRes-KGA.
external <- predictions %>%
  filter(strategy != "combined_cv", model == "cox_residual_kg_attention_nohazard")
calibration <- list()
for (strategy_name in unique(external$strategy)) {
  item <- external %>% filter(strategy == strategy_name)
  fit <- survfit(Surv(time, event) ~ 1, data = item)
  sm <- summary(fit, times = c(12, 24, 36, 60), extend = TRUE)
  label <- ifelse(strategy_name == "train_CGGA_test_TCGA", "CGGA to TCGA", "TCGA to CGGA")
  for (horizon in c(12, 24, 36, 60)) {
    calibration[[length(calibration) + 1]] <- data.frame(
      validation = label,
      horizon = horizon,
      predicted = mean(item[[paste0("survival_", horizon)]], na.rm = TRUE),
      observed = sm$surv[which(sm$time == horizon)[1]]
    )
  }
}
calibration <- bind_rows(calibration)
p7 <- ggplot(calibration, aes(observed, predicted, color = factor(horizon))) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "#777777") +
  geom_point(size = 3) +
  geom_line(aes(group = validation), alpha = 0.4) +
  facet_wrap(~validation) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_color_brewer(palette = "Dark2", name = "Month") +
  labs(
    title = "External calibration of CoxRes-KGA",
    subtitle = "Mean predicted survival compared with Kaplan-Meier survival",
    x = "Observed survival", y = "Mean predicted survival"
  ) +
  theme_report()
save_plot("figure_07_external_calibration.png", p7, 9.2, 4.8)

# Figure 8: risk follows known grade and molecular patterns.
subgroups <- internal %>%
  mutate(
    grade_label = factor(grade, levels = c(2, 3, 4), labels = c("Grade 2", "Grade 3", "Grade 4")),
    idh_label = ifelse(idh_mutant == 1, "IDH mutant", "IDH wildtype"),
    mgmt_label = ifelse(mgmt_methylated == 1, "MGMT methylated", "MGMT unmethylated")
  )
g1 <- ggplot(subgroups, aes(grade_label, risk_score, fill = grade_label)) +
  geom_boxplot(outlier.alpha = 0.15) + facet_wrap(~dataset) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "WHO grade", x = NULL, y = "Predicted risk") +
  theme_report() + theme(legend.position = "none")
g2 <- ggplot(subgroups, aes(idh_label, risk_score, fill = idh_label)) +
  geom_boxplot(outlier.alpha = 0.15) + facet_wrap(~dataset) +
  scale_fill_manual(values = c("#4C78A8", "#B4463A")) +
  labs(title = "IDH status", x = NULL, y = "Predicted risk") +
  theme_report() + theme(legend.position = "none")
g3 <- ggplot(subgroups, aes(mgmt_label, risk_score, fill = mgmt_label)) +
  geom_boxplot(outlier.alpha = 0.15) + facet_wrap(~dataset) +
  scale_fill_manual(values = c("#59A14F", "#E15759")) +
  labs(title = "MGMT status", x = NULL, y = "Predicted risk") +
  theme_report() + theme(legend.position = "none", axis.text.x = element_text(angle = 15, hjust = 1))
save_plot(
  "figure_08_subgroup_risk.png",
  g1 / g2 / g3 + plot_annotation(
    title = "Knowledge-guided model risk across clinical subgroups",
    theme = theme(plot.title = element_text(face = "bold", family = "Arial", color = "#17365D", size = 14))
  ),
  10.2, 12.0
)

# Figure 9: integrated model-performance scorecard.
scorecard_model_order <- c(
  "CoxRes-KGA",
  "Random survival forest",
  "Survival SVM",
  "XGBoost AFT",
  "Regular neural network",
  "Linear regression"
)
scorecard <- performance %>%
  select(validation, model_label, c_index, auc_mean, ibs) %>%
  pivot_longer(
    c(c_index, auc_mean, ibs),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    validation = factor(
      validation,
      levels = c(
        "Internal combined 5-fold CV",
        "External CGGA to TCGA",
        "External TCGA to CGGA"
      )
    ),
    metric = factor(
      metric,
      levels = c("c_index", "auc_mean", "ibs"),
      labels = c("C-index \u2191", "Mean AUC \u2191", "IBS \u2193")
    ),
    model_label = factor(model_label, levels = rev(scorecard_model_order)),
    value_label = sprintf("%.3f", value)
  )

p9 <- ggplot(scorecard, aes(value, model_label, color = model_label)) +
  geom_point(size = 2.7) +
  geom_text(
    aes(label = value_label),
    color = "#333333", hjust = -0.35, size = 2.55,
    show.legend = FALSE
  ) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(values = model_colors, guide = "none") +
  scale_x_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = c(0.06, 0.25))
  ) +
  labs(
    title = "Model performance scorecard across validation settings",
    subtitle = "Exact estimates are shown beside each point; arrows indicate the preferred direction",
    x = NULL, y = NULL
  ) +
  theme_report() +
  theme(
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.grid.major.x = element_line(color = "#EEF1F4", linewidth = 0.35),
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    strip.text = element_text(face = "bold", color = "#17365D", size = 9),
    axis.text.y = element_text(color = "#333333", size = 8.2),
    axis.text.x = element_text(color = "#666666", size = 7.5),
    panel.spacing = unit(0.65, "lines"),
    plot.margin = margin(8, 18, 8, 8)
  )
save_plot("figure_09_performance_scorecard.png", p9, 12.0, 8.8)

# Figure 10: bootstrap confidence intervals for all reported metrics.
bootstrap_performance <- read.csv(
  file.path(selected_root, "metrics", "bootstrap_performance.csv"),
  check.names = FALSE
) %>%
  mutate(
    model_label = factor(unname(model_labels[model]), levels = rev(scorecard_model_order)),
    validation = factor(
      validation,
      levels = c(
        "Internal combined 5-fold CV",
        "External CGGA to TCGA",
        "External TCGA to CGGA"
      )
    ),
    metric = factor(
      metric,
      levels = c("c_index", "auc_mean", "ibs"),
      labels = c("C-index \u2191", "Mean AUC \u2191", "Reported IBS \u2193")
    )
  )

p10 <- ggplot(
  bootstrap_performance,
  aes(estimate, model_label, color = model_label)
) +
  geom_errorbar(
    aes(xmin = ci_low, xmax = ci_high),
    orientation = "y", width = 0, linewidth = 0.7, alpha = 0.8
  ) +
  geom_point(size = 2.5) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(values = model_colors, guide = "none") +
  scale_x_continuous(labels = function(x) sprintf("%.2f", x)) +
  labs(
    title = "Bootstrap uncertainty across models and validation settings",
    subtitle = "Points are corrected-cohort estimates; bars are cohort-stratified paired-bootstrap 95% confidence intervals",
    x = NULL, y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    axis.text.y = element_text(size = 8),
    panel.spacing = unit(0.7, "lines")
  )
save_plot("figure_10_bootstrap_intervals.png", p10, 12.0, 8.8)

# Figure 11: paired bootstrap advantage of CoxRes-KGA over each comparator.
paired_advantage <- read.csv(
  file.path(selected_root, "metrics", "paired_bootstrap_advantage.csv"),
  check.names = FALSE
) %>%
  mutate(
    comparator_label = factor(
      unname(model_labels[comparator]),
      levels = rev(setdiff(scorecard_model_order, "CoxRes-KGA"))
    ),
    validation = factor(
      validation,
      levels = c(
        "Internal combined 5-fold CV",
        "External CGGA to TCGA",
        "External TCGA to CGGA"
      )
    ),
    metric = factor(
      metric,
      levels = c("c_index", "auc_mean", "ibs"),
      labels = c("C-index", "Mean AUC", "Reported IBS")
    ),
    interval_excludes_zero = ci_low > 0 | ci_high < 0
  )

p11 <- ggplot(
  paired_advantage,
  aes(advantage, comparator_label, color = interval_excludes_zero)
) +
  geom_vline(xintercept = 0, color = "#77838F", linewidth = 0.55, linetype = 2) +
  geom_errorbar(
    aes(xmin = ci_low, xmax = ci_high),
    orientation = "y", width = 0, linewidth = 0.7
  ) +
  geom_point(size = 2.5) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(
    values = c(`FALSE` = "#6F91A6", `TRUE` = "#C7352D"),
    labels = c(`FALSE` = "95% CI includes zero", `TRUE` = "95% CI excludes zero"),
    name = NULL
  ) +
  labs(
    title = "Paired performance advantage of CoxRes-KGA",
    subtitle = "Positive values favor CoxRes-KGA; IBS is direction-adjusted so lower prediction error maps to a positive advantage",
    x = "Direction-adjusted paired difference", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    axis.text.y = element_text(size = 8),
    legend.position = "bottom",
    panel.spacing = unit(0.7, "lines")
  )
save_plot("figure_11_paired_advantage.png", p11, 12.0, 8.8)

# Figure 12: calibration-in-the-large error for every model and horizon.
calibration_all <- list()
strategy_labels <- c(
  combined_cv = "Internal combined 5-fold CV",
  train_CGGA_test_TCGA = "External CGGA to TCGA",
  train_TCGA_test_CGGA = "External TCGA to CGGA"
)
for (strategy_name in names(strategy_labels)) {
  outcome_data <- predictions %>%
    filter(strategy == strategy_name, model == "cox_residual_kg_attention_nohazard")
  fit <- survfit(Surv(time, event) ~ 1, data = outcome_data)
  observed_summary <- summary(fit, times = c(12, 24, 36, 60), extend = TRUE)
  observed <- setNames(observed_summary$surv, observed_summary$time)
  for (model_name in names(model_labels)) {
    item <- predictions %>% filter(strategy == strategy_name, model == model_name)
    for (horizon in c(12, 24, 36, 60)) {
      predicted_value <- mean(item[[paste0("survival_", horizon)]], na.rm = TRUE)
      observed_value <- unname(observed[as.character(horizon)])
      calibration_all[[length(calibration_all) + 1]] <- data.frame(
        validation = strategy_labels[[strategy_name]],
        model_label = unname(model_labels[[model_name]]),
        horizon = paste0(horizon, " mo"),
        predicted = predicted_value,
        observed = observed_value,
        calibration_error = predicted_value - observed_value
      )
    }
  }
}
calibration_all <- bind_rows(calibration_all) %>%
  mutate(
    validation = factor(validation, levels = unname(strategy_labels)),
    model_label = factor(model_label, levels = rev(scorecard_model_order)),
    horizon = factor(horizon, levels = c("12 mo", "24 mo", "36 mo", "60 mo")),
    error_label = sprintf("%+.2f", calibration_error)
  )

p12 <- ggplot(calibration_all, aes(horizon, model_label, fill = calibration_error)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = error_label), size = 2.7, color = "#263238") +
  facet_wrap(~validation, ncol = 1) +
  scale_fill_gradient2(
    low = "#3B75AF", mid = "#F7F7F7", high = "#C7352D", midpoint = 0,
    name = "Predicted \u2212\nobserved"
  ) +
  labs(
    title = "All-model calibration-in-the-large",
    subtitle = "Cell values are mean predicted survival minus Kaplan-Meier survival; values near zero indicate better agreement",
    x = "Prediction horizon", y = NULL
  ) +
  theme_report() +
  theme(
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    axis.text.y = element_text(size = 8.2),
    legend.position = "right",
    panel.spacing = unit(0.75, "lines")
  )
save_plot("figure_12_all_model_calibration.png", p12, 9.4, 9.4)

# Figure 13: Efron/learnable-bias sensitivity analysis retained as a non-primary result.
sensitivity_dir <- file.path(root, "results", "sensitivity", "efron_ablation", "metrics")
delta <- read.csv(file.path(sensitivity_dir, "performance_delta.csv"), check.names = FALSE) %>%
  transmute(
    validation,
    `C-index` = delta_c_index,
    `Mean AUC` = delta_auc_mean,
    `Reported IBS` = -delta_ibs
  ) %>%
  pivot_longer(c(`C-index`, `Mean AUC`, `Reported IBS`), names_to = "metric", values_to = "adjusted_delta") %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    direction = ifelse(adjusted_delta >= 0, "Ablation favored", "Primary favored")
  )
frequency <- read.csv(file.path(sensitivity_dir, "selection_frequency.csv"), check.names = FALSE) %>%
  mutate(selected_config = gsub("_", " ", selected_config))

p13a <- ggplot(delta, aes(adjusted_delta, validation, color = direction)) +
  geom_vline(xintercept = 0, linetype = 2, color = "#87939D") +
  geom_segment(aes(x = 0, xend = adjusted_delta, yend = validation), linewidth = 0.8) +
  geom_point(size = 2.8) +
  geom_text(aes(label = sprintf("%+.4f", adjusted_delta)), hjust = ifelse(delta$adjusted_delta >= 0, -0.15, 1.15), size = 2.7, show.legend = FALSE) +
  facet_wrap(~metric, scales = "free_x", nrow = 1) +
  scale_x_continuous(expand = expansion(mult = c(0.30, 0.30))) +
  scale_color_manual(values = c("Ablation favored" = "#287D6B", "Primary favored" = "#B53A32")) +
  labs(
    title = "Efron and learnable-bias sensitivity analysis",
    subtitle = "Direction-adjusted change from the primary CoxRes-KGA; positive values favor the ablation",
    x = "Ablation minus primary (IBS direction reversed)", y = NULL, color = NULL
  ) +
  theme_report() +
  theme(strip.background = element_rect(fill = "#EEF3F8", color = NA), panel.spacing = unit(0.8, "lines"))

p13b <- ggplot(frequency, aes(jobs_selected, reorder(selected_config, jobs_selected))) +
  geom_col(width = 0.58, fill = "#557A95") +
  geom_text(aes(label = jobs_selected), hjust = -0.25, size = 3) +
  scale_x_continuous(breaks = 0:7, limits = c(0, max(frequency$jobs_selected) + 0.8)) +
  labs(title = "Training-only configuration selections", x = "Validation jobs selected", y = NULL) +
  theme_report() +
  theme(panel.grid.major.y = element_blank(), legend.position = "none")

save_plot("figure_13_efron_sensitivity.png", p13a / p13b + plot_layout(heights = c(2.2, 1)), 10.2, 8.0)

# Figure 14: controlled architecture sensitivity with paired uncertainty.
architecture_comparison <- read.csv(
  file.path(
    root, "results", "sensitivity", "architecture_comparison",
    "paired_bootstrap_differences.csv"
  ),
  check.names = FALSE
)
architecture_comparison <- bind_rows(
  architecture_comparison %>% transmute(
    validation, architecture, metric = "C-index",
    estimate = delta_c_index, low = delta_c_index_low, high = delta_c_index_high
  ),
  architecture_comparison %>% transmute(
    validation, architecture, metric = "Mean AUC",
    estimate = delta_auc_mean, low = delta_auc_mean_low, high = delta_auc_mean_high
  ),
  architecture_comparison %>% transmute(
    validation, architecture, metric = "Reported IBS",
    estimate = -delta_ibs, low = -delta_ibs_high, high = -delta_ibs_low
  )
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    architecture = factor(architecture, levels = c("16d/1h", "16d/2h")),
    label = sprintf("%+.4f", estimate),
    label_x = ifelse(estimate >= 0, high, low),
    label_hjust = ifelse(estimate >= 0, -0.12, 1.12)
  )

p14 <- ggplot(
  architecture_comparison,
  aes(estimate, architecture, color = architecture)
) +
  geom_vline(xintercept = 0, color = "#7D8993", linetype = 2, linewidth = 0.55) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.8) +
  geom_point(size = 2.8) +
  geom_text(aes(x = label_x, label = label, hjust = label_hjust), size = 2.7, show.legend = FALSE) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(values = c("16d/1h" = "#266F8E", "16d/2h" = "#C26A35"), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0.24, 0.34))) +
  labs(
    title = "CoxRes-KGA architecture sensitivity",
    subtitle = "Paired difference from 32d/1-head; positive values favor the alternative (IBS direction reversed)",
    x = "Direction-adjusted paired difference with 95% bootstrap CI", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing = unit(0.75, "lines"),
    legend.position = "bottom"
  )
save_plot("figure_14_architecture_sensitivity.png", p14, 11.0, 7.4)

# Figure 15: structured improvement variants compared with retained 16d/1-head.
structured <- read.csv(
  file.path(
    root, "results", "sensitivity", "structured_improvement_comparison",
    "paired_bootstrap_differences.csv"
  ),
  check.names = FALSE
)
structured <- bind_rows(
  structured %>% transmute(
    validation, variant, metric = "C-index",
    estimate = delta_c_index, low = delta_c_index_low, high = delta_c_index_high
  ),
  structured %>% transmute(
    validation, variant, metric = "Mean AUC",
    estimate = delta_auc_mean, low = delta_auc_mean_low, high = delta_auc_mean_high
  ),
  structured %>% transmute(
    validation, variant, metric = "Reported IBS",
    estimate = -delta_ibs, low = -delta_ibs_high, high = -delta_ibs_low
  )
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    variant = recode(
      variant,
      `A: nonlinear Cox` = "A: nonlinear Cox",
      `B: nonlinear + crossfit` = "B: + cross-fit",
      `C: nonlinear + crossfit + orthogonal` = "C: + orthogonality"
    ),
    variant = factor(variant, levels = c("A: nonlinear Cox", "B: + cross-fit", "C: + orthogonality")),
    conclusion = ifelse(low > 0 | high < 0, "95% CI excludes zero", "95% CI includes zero")
  )

p15 <- ggplot(structured, aes(estimate, variant, color = conclusion)) +
  geom_vline(xintercept = 0, color = "#7D8993", linetype = 2, linewidth = 0.55) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.8) +
  geom_point(size = 2.7) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(
    values = c("95% CI includes zero" = "#5D7F95", "95% CI excludes zero" = "#B5453C"),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "Structured CoxRes-KGA improvement experiments",
    subtitle = "Paired differences from retained 16d/1-head; positive values favor the variant (IBS direction reversed)",
    x = "Direction-adjusted paired difference with 95% bootstrap CI", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing = unit(0.75, "lines"),
    axis.text.y = element_text(size = 7.8),
    legend.position = "bottom"
  )
save_plot("figure_15_structured_improvements.png", p15, 11.0, 8.2)

# Figure 16: full-cohort versus primary-tumor-only six-model sensitivity.
primary_comparison <- read.csv(
  file.path(
    root, "results", "sensitivity", "primary_tumor_only", "metrics",
    "full_vs_primary_summary.csv"
  ),
  check.names = FALSE
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    model_label = factor(unname(model_labels[model]), levels = rev(comparison_order))
  )
primary_long <- bind_rows(
  primary_comparison %>% transmute(validation, model_label, metric = "C-index", Full = c_index_full, `Primary tumors only` = c_index_primary),
  primary_comparison %>% transmute(validation, model_label, metric = "Mean AUC", Full = auc_mean_full, `Primary tumors only` = auc_mean_primary),
  primary_comparison %>% transmute(validation, model_label, metric = "Reported IBS", Full = ibs_full, `Primary tumors only` = ibs_primary)
) %>%
  pivot_longer(c(Full, `Primary tumors only`), names_to = "analysis", values_to = "value") %>%
  mutate(metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")))

p16 <- ggplot(primary_long, aes(value, model_label, color = analysis, group = model_label)) +
  geom_line(color = "#C8D0D6", linewidth = 0.7) +
  geom_point(size = 2.5) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(values = c("Full" = "#8A99A5", "Primary tumors only" = "#246B8E"), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0.12, 0.12))) +
  labs(
    title = "Primary-tumor-only sensitivity across all six models",
    subtitle = "Connected points compare the original full cohorts with the primary-tumor subset; lower IBS is preferred",
    x = "Metric value", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing = unit(0.75, "lines"),
    axis.text.y = element_text(size = 7.7),
    legend.position = "bottom"
  )
save_plot("figure_16_primary_tumor_sensitivity.png", p16, 11.0, 8.2)

# Figure 17: paired selected dropout-0.40/Adam comparison with the reference model.
dropout_adam <- read.csv(
  file.path(
    root, "results", "sensitivity", "dropout040_adam", "metrics",
    "retained_comparison.csv"
  ),
  check.names = FALSE
)
dropout_adam_long <- bind_rows(
  dropout_adam %>% transmute(
    validation, metric = "C-index", estimate = delta_c_index,
    low = delta_c_index_low, high = delta_c_index_high
  ),
  dropout_adam %>% transmute(
    validation, metric = "Mean AUC", estimate = delta_auc_mean,
    low = delta_auc_mean_low, high = delta_auc_mean_high
  ),
  dropout_adam %>% transmute(
    validation, metric = "Reported IBS", estimate = -delta_ibs,
    low = -delta_ibs_high, high = -delta_ibs_low
  )
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    conclusion = ifelse(low > 0 | high < 0, "95% CI excludes zero", "95% CI includes zero")
  )

p17 <- ggplot(dropout_adam_long, aes(estimate, validation, color = conclusion)) +
  geom_vline(xintercept = 0, color = "#7D8993", linetype = 2, linewidth = 0.55) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.9) +
  geom_point(size = 3.0) +
  facet_wrap(~metric, scales = "free_x", nrow = 1) +
  scale_color_manual(
    values = c("95% CI includes zero" = "#5D7F95", "95% CI excludes zero" = "#B5453C"),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "Selection of dropout 0.40 with Adam",
    subtitle = "Paired difference from dropout-0.12/AdamW reference; positive values favor the selected configuration",
    x = "Direction-adjusted paired difference with 95% bootstrap CI", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing.x = unit(2.5, "lines"),
    axis.text.x = element_text(size = 7.5),
    legend.position = "bottom"
  )
save_plot("figure_17_dropout_adam_sensitivity.png", p17, 12.0, 4.8)

# Figure 18: three-configuration training and attention comparison.
training_variants <- read.csv(
  file.path(
    root, "results", "sensitivity", "dropout050_adamw_h2", "metrics",
    "three_configuration_points.csv"
  ),
  check.names = FALSE
)
training_variants_long <- bind_rows(
  training_variants %>% transmute(validation, configuration, metric = "C-index", value = c_index),
  training_variants %>% transmute(validation, configuration, metric = "Mean AUC", value = auc_mean),
  training_variants %>% transmute(validation, configuration, metric = "Reported IBS", value = ibs)
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    configuration = recode(
      configuration,
      "Retained 0.12/AdamW/1h" = "Reference 0.12/AdamW/1h",
      "Best-rank 0.40/Adam/1h" = "Selected 0.40/Adam/1h"
    ),
    configuration = factor(
      configuration,
      levels = c(
        "Reference 0.12/AdamW/1h",
        "Selected 0.40/Adam/1h",
        "Candidate 0.50/AdamW/2h"
      )
    )
  )

p18 <- ggplot(training_variants_long, aes(value, configuration, color = configuration)) +
  geom_line(aes(group = validation), color = "#D3DADF", linewidth = 0.65) +
  geom_point(size = 2.8) +
  facet_grid(validation ~ metric, scales = "free_x") +
  scale_color_manual(
    values = c(
      "Reference 0.12/AdamW/1h" = "#8A99A5",
      "Selected 0.40/Adam/1h" = "#246B8E",
      "Candidate 0.50/AdamW/2h" = "#B65A45"
    ),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "CoxRes-KGA training-configuration selection",
    subtitle = "Three configurations on identical validation samples; lower reported IBS is preferred",
    x = "Metric value", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing = unit(0.9, "lines"),
    axis.text.y = element_text(size = 7.5),
    legend.position = "bottom"
  )
save_plot("figure_18_training_configuration_sensitivity.png", p18, 11.5, 8.0)

# Figure 19: single-factor dropout 0.40 versus 0.60 comparison.
dropout060 <- read.csv(
  file.path(
    root, "results", "sensitivity", "dropout060_adam", "metrics",
    "dropout040_vs_dropout060.csv"
  ),
  check.names = FALSE
)
dropout060_long <- bind_rows(
  dropout060 %>% transmute(validation, metric = "C-index", estimate = delta_c_index, low = delta_c_index_low, high = delta_c_index_high),
  dropout060 %>% transmute(validation, metric = "Mean AUC", estimate = delta_auc_mean, low = delta_auc_mean_low, high = delta_auc_mean_high),
  dropout060 %>% transmute(validation, metric = "Reported IBS", estimate = -delta_ibs, low = -delta_ibs_high, high = -delta_ibs_low)
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    conclusion = ifelse(low > 0 | high < 0, "95% CI excludes zero", "95% CI includes zero")
  )

p19 <- ggplot(dropout060_long, aes(estimate, validation, color = conclusion)) +
  geom_vline(xintercept = 0, color = "#7D8993", linetype = 2, linewidth = 0.55) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.9) +
  geom_point(size = 3.0) +
  facet_wrap(~metric, scales = "free_x", nrow = 1) +
  scale_color_manual(
    values = c("95% CI includes zero" = "#5D7F95", "95% CI excludes zero" = "#B5453C"),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "Single-factor dropout sensitivity",
    subtitle = "Dropout 0.60 minus 0.40 with Adam and one head fixed; positive values favor dropout 0.60",
    x = "Direction-adjusted paired difference with 95% bootstrap CI", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing.x = unit(2.5, "lines"),
    axis.text.x = element_text(size = 7.5),
    legend.position = "bottom"
  )
save_plot("figure_19_dropout060_sensitivity.png", p19, 12.0, 4.8)

# Figure 20: controlled one-head dropout 0.50/AdamW comparison.
dropout050_adamw_h1 <- read.csv(
  file.path(
    root, "results", "sensitivity", "dropout050_adamw_h1", "metrics",
    "dropout040_adam_vs_dropout050_adamw_h1.csv"
  ),
  check.names = FALSE
)
dropout050_adamw_h1_long <- bind_rows(
  dropout050_adamw_h1 %>% transmute(validation, metric = "C-index", estimate = delta_c_index, low = delta_c_index_low, high = delta_c_index_high),
  dropout050_adamw_h1 %>% transmute(validation, metric = "Mean AUC", estimate = delta_auc_mean, low = delta_auc_mean_low, high = delta_auc_mean_high),
  dropout050_adamw_h1 %>% transmute(validation, metric = "Reported IBS", estimate = -delta_ibs, low = -delta_ibs_high, high = -delta_ibs_low)
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    conclusion = ifelse(low > 0 | high < 0, "95% CI excludes zero", "95% CI includes zero")
  )

p20 <- ggplot(dropout050_adamw_h1_long, aes(estimate, validation, color = conclusion)) +
  geom_vline(xintercept = 0, color = "#7D8993", linetype = 2, linewidth = 0.55) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.9) +
  geom_point(size = 3.0) +
  facet_wrap(~metric, scales = "free_x", nrow = 1) +
  scale_color_manual(
    values = c("95% CI includes zero" = "#5D7F95", "95% CI excludes zero" = "#B5453C"),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "Dropout and optimizer sensitivity with architecture fixed",
    subtitle = "Dropout 0.50/AdamW minus dropout 0.40/Adam; positive values favor dropout 0.50/AdamW",
    x = "Direction-adjusted paired difference with 95% bootstrap CI", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing.x = unit(2.5, "lines"),
    axis.text.x = element_text(size = 7.5),
    legend.position = "bottom"
  )
save_plot("figure_20_dropout050_adamw_h1_sensitivity.png", p20, 12.0, 4.8)

# Figure 21: 32-dimensional/120-epoch capacity-duration comparison.
d32_e120 <- read.csv(
  file.path(
    root, "results", "sensitivity", "d32_e120_adam", "metrics",
    "d16_e170_vs_d32_e120.csv"
  ),
  check.names = FALSE
)
d32_e120_long <- bind_rows(
  d32_e120 %>% transmute(validation, metric = "C-index", estimate = delta_c_index, low = delta_c_index_low, high = delta_c_index_high),
  d32_e120 %>% transmute(validation, metric = "Mean AUC", estimate = delta_auc_mean, low = delta_auc_mean_low, high = delta_auc_mean_high),
  d32_e120 %>% transmute(validation, metric = "Reported IBS", estimate = -delta_ibs, low = -delta_ibs_high, high = -delta_ibs_low)
) %>%
  mutate(
    validation = factor(validation, levels = validation_order),
    metric = factor(metric, levels = c("C-index", "Mean AUC", "Reported IBS")),
    conclusion = ifelse(low > 0 | high < 0, "95% CI excludes zero", "95% CI includes zero")
  )

p21 <- ggplot(d32_e120_long, aes(estimate, validation, color = conclusion)) +
  geom_vline(xintercept = 0, color = "#7D8993", linetype = 2, linewidth = 0.55) +
  geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = 0, linewidth = 0.9) +
  geom_point(size = 3.0) +
  facet_wrap(~metric, scales = "free_x", nrow = 1) +
  scale_color_manual(
    values = c("95% CI includes zero" = "#5D7F95", "95% CI excludes zero" = "#B5453C"),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "Representation dimension and training-duration sensitivity",
    subtitle = "32d/120 epochs minus 16d/170 epochs; positive values favor 32d/120 epochs",
    x = "Direction-adjusted paired difference with 95% bootstrap CI", y = NULL
  ) +
  theme_report() +
  theme(
    strip.background = element_rect(fill = "#EEF3F8", color = NA),
    panel.grid.major.y = element_line(color = "#E7EBEF", linewidth = 0.35),
    panel.spacing.x = unit(2.5, "lines"),
    axis.text.x = element_text(size = 7.5),
    legend.position = "bottom"
  )
save_plot("figure_21_d32_e120_sensitivity.png", p21, 12.0, 4.8)

writeLines(capture.output(sessionInfo()), file.path(root, "logs", "R_session_info.txt"))
cat("Figures written to", fig_dir, "\n")
