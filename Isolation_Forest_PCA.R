###############################################################################
######################## PCA + ISOLATION FOREST BASELINE ######################
###############################################################################

library(pbapply)
library(isotree)   # Isolation Forest (reines R)
library(pROC)

# Wir verwenden compute_logmel_context aus deinem AE-Code
# NICHT ändern!

# Parameter
pca_var_explained <- 0.98   # Ziel: 98% Varianz
n_trees <- 200              # IF Bäume

results_pca_if <- list()

for (id in unique_ids) {
  
  cat("\n==============================\n")
  cat("PCA + IF – MACHINE ID:", id, "\n")
  cat("==============================\n")
  
  # Dateien pro ID
  train_idx <- which(train_ids == id)
  test_idx  <- which(test_ids  == id)
  
  train_feat <- pblapply(train_files[train_idx], compute_logmel_context)
  test_feat  <- pblapply(test_files[test_idx],  compute_logmel_context)
  
  train_feat <- train_feat[!sapply(train_feat, is.null)]
  test_feat  <- test_feat[!sapply(test_feat,  is.null)]
  
  if (length(train_feat) == 0 || length(test_feat) == 0) {
    cat(">> Keine gültigen Features für ID", id, " – übersprungen.\n")
    next
  }
  
  # Stack
  X_train <- do.call(rbind, train_feat)
  X_test  <- do.call(rbind, test_feat)
  
  # Clip-Labels (normal/anomaly)
  y_test_clip <- ifelse(grepl("anomaly",
                              basename(test_files[test_idx]),
                              ignore.case = TRUE),
                        1L, 0L)
  
  frame_counts <- sapply(test_feat, nrow)
  test_groups  <- rep(seq_along(frame_counts), times = frame_counts)
  
  ######################################################################
  # 1) PCA FIT AUF NUR NORMALEN TRAININGSFRAMES — DCASE-KONFORM
  ######################################################################
  
  pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
  
  # Varianzschwelle
  var_explained <- cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2)
  k <- which(var_explained >= pca_var_explained)[1]
  
  cat("PCA-Komponenten:", k, "\n")
  
  # PCA-Projektion
  Z_train <- pca_model$x[, 1:k]
  Z_test  <- scale(X_test,
                   center = pca_model$center,
                   scale  = pca_model$scale) %*% pca_model$rotation[, 1:k]
  
  ######################################################################
  # 2) ISOLATION FOREST TRAINING (nur normal)
  ######################################################################
  
  if_model <- isolation.forest(
    Z_train,
    ntrees = n_trees,
    sample_size = 256,
    ndim = 1,
    seed = 123
  )
  
  ######################################################################
  # 3) Scores berechnen
  ######################################################################
  
  frame_scores <- predict(if_model, Z_test, type = "score")
  
  # Clip-Level Score = Durchschnitt über Frames
  clip_scores <- tapply(frame_scores, test_groups, mean)
  
  ######################################################################
  # 4) AUC & pAUC
  ######################################################################
  
  auc_val  <- pROC::auc(y_test_clip, clip_scores)
  pauc_val <- pROC::auc(
    y_test_clip,
    clip_scores,
    partial.auc = c(1, 0.9),
    partial.auc.focus = "specificity"
  )
  
  cat("AUC:  ", as.numeric(auc_val),  "\n")
  cat("pAUC: ", as.numeric(pauc_val), "\n")
  
  results_pca_if[[id]] <- list(
    auc    = as.numeric(auc_val),
    pauc   = as.numeric(pauc_val),
    scores = clip_scores,
    labels = y_test_clip,
    k_pca  = k
  )
}

###############################################################################
# AUCS ANZEIGEN
###############################################################################

sapply(results_pca_if, function(x) x$auc)
#00        02        04        06 
#0.5246683 0.7071588 0.5316092 0.8355956 
mean(sapply(results_pca_if, function(x) x$auc))
#[1] 0.649758
