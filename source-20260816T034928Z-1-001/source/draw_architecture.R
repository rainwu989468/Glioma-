draw_coxres_kga <- function(output_path) {
  grDevices::png(output_path, width = 6600, height = 3650, res = 300, bg = "white")
  on.exit(grDevices::dev.off(), add = TRUE)
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(xscale = c(0, 24), yscale = c(0, 13.2)))

  u <- function(x) grid::unit(x, "native")
  pal <- list(
    ink = "#17212B", muted = "#66717D", line = "#36414C", grid = "#B9C2CA",
    sand = "#F7EFE4", sand_edge = "#8A735E",
    blue = "#2E67C7", blue_fill = "#EEF4FF", blue_side = "#C8D8F4",
    orange = "#E66A25", orange_fill = "#FFF1E7", orange_side = "#F5C3A4",
    green = "#3E8B57", green_fill = "#EEF8F0", green_side = "#BFDEC7",
    purple = "#7451B5", purple_fill = "#F5F0FC", purple_side = "#D5C7ED",
    gold = "#C78B12", gold_fill = "#FFF7DA",
    rose = "#C84D67", rose_fill = "#FFF0F3"
  )

  txt <- function(text, x, y, size = 12, face = "plain", color = pal$ink,
                  just = "centre", rot = 0, lineheight = 0.92) {
    grid::grid.text(
      text, x = u(x), y = u(y), just = just, rot = rot,
      gp = grid::gpar(
        col = color, fontsize = size, fontface = face,
        fontfamily = "Arial", lineheight = lineheight
      )
    )
  }

  rect <- function(x, y, w, h, text = "", fill = "white", border = pal$line,
                   size = 11, face = "plain", lwd = 1.1, lty = 1, radius = 0.06,
                   color = pal$ink) {
    grid::grid.roundrect(
      x = u(x), y = u(y), width = u(w), height = u(h),
      r = grid::unit(radius, "snpc"),
      gp = grid::gpar(fill = fill, col = border, lwd = lwd, lty = lty)
    )
    if (nzchar(text)) txt(text, x, y, size, face, color)
  }

  slab <- function(x, y, w, h, text, fill, side, border,
                   size = 11, face = "plain", depth = 0.18) {
    grid::grid.polygon(
      x = u(c(x - w / 2, x + w / 2, x + w / 2 + depth, x - w / 2 + depth)),
      y = u(c(y + h / 2, y + h / 2, y + h / 2 + depth, y + h / 2 + depth)),
      gp = grid::gpar(fill = side, col = border, lwd = 0.9)
    )
    grid::grid.polygon(
      x = u(c(x + w / 2, x + w / 2 + depth, x + w / 2 + depth, x + w / 2)),
      y = u(c(y - h / 2, y - h / 2 + depth, y + h / 2 + depth, y + h / 2)),
      gp = grid::gpar(fill = side, col = border, lwd = 0.9)
    )
    grid::grid.rect(
      x = u(x), y = u(y), width = u(w), height = u(h),
      gp = grid::gpar(fill = fill, col = border, lwd = 1.0)
    )
    txt(text, x, y, size, face)
  }

  line <- function(x, y, color = pal$line, lwd = 1.2, dashed = FALSE,
                   arrow_end = FALSE) {
    grid::grid.lines(
      x = u(x), y = u(y),
      arrow = if (arrow_end) grid::arrow(length = grid::unit(0.085, "inches"), type = "closed") else NULL,
      gp = grid::gpar(
        col = color, lwd = lwd, lty = if (dashed) "22" else "solid",
        lineend = "round", linejoin = "round"
      )
    )
  }

  arrow <- function(x1, y1, x2, y2, color = pal$line, lwd = 1.35, dashed = FALSE) {
    line(c(x1, x2), c(y1, y2), color, lwd, dashed, TRUE)
  }

  plus <- function(x, y, r = 0.18) {
    grid::grid.circle(x = u(x), y = u(y), r = u(r),
                      gp = grid::gpar(fill = "white", col = pal$line, lwd = 1.0))
    txt("+", x, y - 0.01, 13, "bold")
  }

  group_box <- function(x, y, w, h, title, border = pal$grid, fill = "white") {
    rect(x, y, w, h, fill = fill, border = border, lwd = 1.1, lty = "22", radius = 0.035)
    txt(title, x - w / 2 + 0.18, y + h / 2 + 0.17, 13, "bold", just = "left")
  }

  # Main pathway -----------------------------------------------------------------
  txt("Clinical and molecular predictors", 0.18, 12.58, 15, "bold", just = "left")
  input_names <- c("Age", "Sex", "WHO grade", "IDH status", "1p/19q codeletion", "MGMT methylation")
  for (back in c(0.24, 0.12)) {
    rect(1.45 - back, 10.15 + back, 2.25, 3.65, fill = "#FBF6EF", border = pal$sand_edge, lwd = 0.7)
  }
  rect(1.45, 10.05, 2.25, 3.65, fill = "#FBF6EF", border = pal$sand_edge, lwd = 0.9)
  for (i in seq_along(input_names)) {
    rect(1.45, 11.38 - (i - 1) * 0.54, 1.92, 0.42, input_names[i], pal$sand,
         pal$sand_edge, 9.4, lwd = 0.75, radius = 0.035)
  }

  group_box(4.20, 10.05, 2.50, 3.60, "Fold-fitted preprocessing")
  prep <- c("Median / mode\nimputation", "Scaling", "Categorical\nencoding", "Missingness\nindicators")
  for (i in seq_along(prep)) {
    slab(4.13, 11.30 - (i - 1) * 0.73, 1.63, 0.48, prep[i], pal$blue_fill,
         pal$blue_side, pal$blue, 8.8)
  }
  arrow(2.62, 10.05, 2.92, 10.05)

  # Cox branch.
  group_box(8.15, 11.24, 4.55, 2.02, "Cox main-effect pathway", pal$orange, "#FFFCFA")
  slab(7.05, 11.17, 1.58, 0.93, "Penalized Cox PH\nfit on training data", pal$orange_fill,
       pal$orange_side, pal$orange, 9.5, "bold")
  slab(9.30, 11.17, 1.60, 0.93, "Standardized Cox\noffset  zCox(x)", pal$orange_fill,
       pal$orange_side, pal$orange, 9.5, "bold")
  arrow(7.95, 11.17, 8.38, 11.17, pal$orange)
  line(c(5.48, 5.82, 5.82, 6.16), c(10.73, 10.73, 11.17, 11.17), pal$line, 1.2, FALSE, TRUE)

  # Neural branch.
  group_box(10.80, 8.50, 9.85, 2.65, "Knowledge-guided neural residual pathway", pal$green, "#FCFEFC")
  slab(6.98, 8.64, 1.58, 1.22, "Transformed scalar\npredictors", pal$green_fill,
       pal$green_side, pal$green, 9.3, "bold")
  txt("Clinical and molecular variables\nplus missingness indicators\nafter fold-fitted preprocessing", 6.98, 7.84, 7.6, color = pal$muted)
  slab(9.18, 8.64, 1.72, 1.22, "Feature tokens\nd = 16", pal$blue_fill,
       pal$blue_side, pal$blue, 10, "bold")
  for (j in 0:4) {
    grid::grid.rect(x = u(8.64 + j * 0.25), y = u(8.15), width = u(0.14), height = u(0.20),
                    gp = grid::gpar(fill = if (j == 0) pal$orange_fill else pal$blue_fill,
                                    col = if (j == 0) pal$orange else pal$blue, lwd = 0.7))
  }
  txt("CLS", 8.66, 7.96, 6.5, color = pal$muted)
  slab(11.48, 8.64, 1.78, 1.22, "Fixed additive\nknowledge bias", pal$gold_fill,
       "#E9D79B", pal$gold, 9.7, "bold")
  slab(13.85, 8.64, 1.90, 1.22, "Transformer encoder\n1 layer · 1 head", pal$green_fill,
       pal$green_side, pal$green, 9.7, "bold")
  slab(16.18, 8.64, 1.64, 1.22, "CLS residual head\n16→16→1 · GELU", pal$green_fill,
       pal$green_side, pal$green, 9.4, "bold")
  line(c(5.48, 5.82, 5.82, 6.08), c(9.37, 9.37, 8.64, 8.64), pal$line, 1.2, FALSE, TRUE)
  arrow(7.86, 8.64, 8.23, 8.64, pal$green)
  arrow(10.14, 8.64, 10.48, 8.64, pal$green)
  arrow(12.47, 8.64, 12.77, 8.64, pal$green)
  arrow(14.90, 8.64, 15.25, 8.64, pal$green)
  line(c(9.30, 9.30, 9.03), c(10.69, 9.55, 9.55), pal$orange, 1.05, TRUE, TRUE)

  # Positive Cox-residual fusion and seed averaging.
  group_box(19.14, 10.02, 2.82, 4.00, "Cox-residual fusion", pal$line, "#FCFDFE")
  rect(19.14, 10.66, 2.32, 1.12,
       "r_seed = z_Cox\n+ softplus(alpha) * r_res", "white", pal$line, 9.4, "bold")
  rect(19.14, 9.12, 2.32, 1.24,
       "Seed-specific risk\ntrained with seeds\n42, 43, 44, 45, and 46", "white", pal$line, 8.3)
  line(c(10.20, 17.42, 17.42, 17.74), c(11.17, 11.17, 10.66, 10.66), pal$orange, 1.25, FALSE, TRUE)
  line(c(17.10, 17.55, 17.55, 17.74), c(8.64, 8.64, 10.25, 10.25), pal$green, 1.25, FALSE, TRUE)
  arrow(19.14, 10.05, 19.14, 9.77)

  slab(22.00, 10.07, 1.65, 1.62, "Continuous\nsurvival risk\nscore  r(x)", pal$blue_fill,
       pal$blue_side, pal$blue, 11, "bold")
  txt("mean of 5 seeds", 22.00, 9.02, 8.3, color = pal$muted)
  arrow(20.58, 9.55, 21.05, 9.90, pal$blue)

  rect(21.02, 7.03, 1.95, 0.92, "Training-derived\nBreslow baseline", pal$purple_fill,
       pal$purple, 9.2, "bold", lty = "22")
  slab(23.15, 7.03, 1.30, 1.08, "Survival\nprobability\nS(t|x)", pal$purple_fill,
       pal$purple_side, pal$purple, 9.6, "bold", depth = 0.12)
  line(c(22.00, 22.00, 21.02), c(9.25, 7.76, 7.50), pal$purple, 1.0, TRUE, TRUE)
  arrow(22.00, 7.03, 22.43, 7.03, pal$purple, 1.1, TRUE)
  txt("12, 24, 36, and 60 months", 22.02, 6.30, 8.0, color = pal$muted)

  # Objective loop.
  grid::grid.curve(
    x1 = u(21.35), y1 = u(9.25), x2 = u(3.20), y2 = u(7.02), curvature = -0.24,
    arrow = grid::arrow(length = grid::unit(0.09, "inches"), type = "closed"),
    gp = grid::gpar(col = pal$line, lwd = 1.15)
  )
  txt("Training objective: Cox partial likelihood + λrank pairwise ranking loss + λres residual L2 penalty",
      11.65, 6.57, 10.5, "bold")

  # Detailed inset ---------------------------------------------------------------
  group_box(6.18, 3.17, 11.45, 5.00, "Expanded knowledge-guided Transformer block", pal$green, "#FCFEFC")
  line(c(12.88, 11.20), c(7.93, 5.67), pal$green, 0.9, TRUE)
  line(c(14.78, 17.25), c(7.93, 5.67), pal$green, 0.9, TRUE)

  slab(1.55, 3.18, 1.20, 1.18, "Feature\ntokens", pal$blue_fill, pal$blue_side, pal$blue, 9.4, "bold", 0.12)
  rect(3.22, 3.18, 1.56, 1.32, "Knowledge bias B\n+0.45 prior pair\n−0.35 other pair", pal$gold_fill,
       pal$gold, 7.8, "bold")
  arrow(2.22, 3.18, 2.42, 3.18, pal$green)

  group_box(8.05, 3.18, 7.48, 3.62, "Transformer encoder layer × 1", pal$grid, "white")
  rect(5.06, 3.18, 1.15, 1.18, "Q  K  V\nlinear\nprojections", pal$blue_fill, pal$blue, 8.2)
  rect(6.58, 3.18, 1.28, 1.18, "Scaled dot-product\nattention\n1 head", pal$blue_fill, pal$blue, 8.2, "bold")
  plus(7.60, 3.18, 0.16)
  rect(8.35, 3.18, 0.90, 1.02, "Add +\nLayerNorm", pal$gold_fill, pal$gold, 8.2)
  rect(9.72, 3.18, 1.25, 1.18, "Feed-forward\n16→48→16\nGELU", pal$green_fill, pal$green, 8.2)
  plus(10.76, 3.18, 0.16)
  rect(11.52, 3.18, 0.90, 1.02, "Add +\nLayerNorm", pal$gold_fill, pal$gold, 8.2)
  arrow(4.00, 3.18, 4.43, 3.18, pal$green)
  arrow(5.66, 3.18, 5.88, 3.18, pal$green)
  arrow(7.27, 3.18, 7.40, 3.18, pal$green)
  arrow(7.78, 3.18, 7.88, 3.18, pal$green)
  arrow(8.82, 3.18, 9.04, 3.18, pal$green)
  arrow(10.37, 3.18, 10.56, 3.18, pal$green)
  arrow(10.94, 3.18, 11.06, 3.18, pal$green)
  arrow(11.98, 3.18, 12.28, 3.18, pal$green)
  line(c(4.03, 4.03, 7.60), c(3.78, 4.26, 4.26), pal$line, 0.9, FALSE, TRUE)
  line(c(8.86, 8.86, 10.76), c(3.78, 4.26, 4.26), pal$line, 0.9, FALSE, TRUE)
  txt("Residual connection", 5.62, 4.46, 7.5, color = pal$muted)
  txt("Residual connection", 9.74, 4.46, 7.5, color = pal$muted)

  # Prespecified priors.
  group_box(16.55, 3.17, 4.45, 5.00, "Prespecified glioma relationships", pal$blue, "#FEFEFF")
  prior_nodes <- data.frame(
    label = c("Age", "Grade", "IDH", "1p/19q", "MGMT"),
    x = c(15.25, 16.55, 15.20, 16.55, 17.90),
    y = c(3.95, 4.43, 2.62, 2.27, 2.62),
    fill = c(pal$gold_fill, pal$green_fill, pal$rose_fill, pal$blue_fill, pal$orange_fill),
    border = c(pal$gold, pal$green, pal$rose, pal$blue, pal$orange)
  )
  edge_idx <- list(c(1,2), c(2,3), c(2,5), c(3,4), c(3,5))
  for (e in edge_idx) {
    line(c(prior_nodes$x[e[1]], prior_nodes$x[e[2]]),
         c(prior_nodes$y[e[1]], prior_nodes$y[e[2]]), pal$line, 0.85, TRUE)
  }
  for (i in seq_len(nrow(prior_nodes))) {
    grid::grid.circle(x = u(prior_nodes$x[i]), y = u(prior_nodes$y[i]), r = u(0.34),
                      gp = grid::gpar(fill = prior_nodes$fill[i], col = prior_nodes$border[i], lwd = 1.0))
    txt(prior_nodes$label[i], prior_nodes$x[i], prior_nodes$y[i], 8.2, "bold")
  }
  txt("Priors define a fixed additive attention bias;\nall token pairs remain available to attention.",
      16.55, 1.18, 8.0, color = pal$muted)

  # Fixed architecture with training-only hyperparameter selection and calibration.
  group_box(21.32, 3.17, 4.25, 5.00, "Fixed architecture; nested tuning", pal$purple, "#FFFEFF")
  rect(21.32, 4.35, 3.52, 0.66, "Architecture fixed for every job", pal$purple_fill, pal$purple, 8.8, "bold")
  rect(21.32, 3.48, 3.52, 0.78, "Token dimension 16\n1 layer · 1 head", pal$purple_fill, pal$purple, 8.6)
  rect(21.32, 2.52, 3.52, 0.78, "3-fold training-only selection of\nCox penalty and loss weights", pal$purple_fill, pal$purple, 8.5)
  rect(21.32, 1.56, 3.52, 0.78, "18% epoch selection; 5-seed refit;\ntraining-derived Breslow calibration", pal$purple_fill, pal$purple, 8.3)
  arrow(21.32, 4.00, 21.32, 3.92, pal$purple, 1.0)
  arrow(21.32, 3.06, 21.32, 2.96, pal$purple, 1.0)
  arrow(21.32, 2.10, 21.32, 2.00, pal$purple, 1.0)

  # Legend and footer.
  arrow(0.70, 0.45, 1.25, 0.45, pal$line, 1.0)
  txt("Forward data flow", 1.42, 0.45, 8.2, just = "left")
  arrow(3.10, 0.45, 3.65, 0.45, pal$line, 1.0, TRUE)
  txt("Prior or training-only flow", 3.82, 0.45, 8.2, just = "left")
  rect(6.55, 0.45, 0.40, 0.22, fill = pal$orange_fill, border = pal$orange, radius = 0.02)
  txt("Cox pathway", 6.83, 0.45, 8.2, just = "left")
  rect(8.42, 0.45, 0.40, 0.22, fill = pal$green_fill, border = pal$green, radius = 0.02)
  txt("Neural pathway", 8.70, 0.45, 8.2, just = "left")
  rect(10.45, 0.45, 0.40, 0.22, fill = pal$purple_fill, border = pal$purple, radius = 0.02)
  txt("Probability calibration", 10.73, 0.45, 8.2, just = "left")
  txt("CoxRes-KGA: Cox-residual knowledge-guided attention network", 23.35, 0.45, 11.5,
      "bold", just = "right")

  grid::popViewport()
  invisible(output_path)
}
