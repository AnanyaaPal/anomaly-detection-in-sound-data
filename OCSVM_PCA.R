###############################################################################
################ One-Class SVM mit PCA – DCASE FAN (pure R) ###################
###############################################################################

library(tuneR)
library(seewave)
library(pbapply)
library(pROC)
library(e1071)   # One-Class SVM

###############################################################################
# 1) Dateien & IDs
###############################################################################

train_dir <- "fan/train"
test_dir  <- "fan/test"

train_files <- list.files(train_dir, pattern="\\.wav$", full.names=TRUE, recursive=TRUE)
test_files  <- list.files(test_dir,  pattern="\\.wav$", full.names=TRUE, recursive=TRUE)

cat("Train:", length(train_files), "\n")
cat("Test: ", length(test_files), "\n")

get_id <- function(x) gsub(".*id_(..)_.*", "\\1", basename(x))
train_ids  <- get_id(train_files)
test_ids   <- get_id(test_files)
unique_ids <- sort(unique(train_ids))

# Clip-Level Labels (für Test)
y_test_all <- ifelse(grepl("anomaly", basename(test_files), ignore.case=TRUE), 1L, 0L)

###############################################################################
# 2) Pooling-Statistik-Features pro Clip (7D), robust
###############################################################################

compute_poolstats <- function(file) {
  w <- tuneR::readWave(file)
  if (w@stereo) w <- mono(w, which = "left")
  
  sp <- seewave::spectro(
    w, f = w@samp.rate, wl = 1024, ovlp = 50,
    plot = FALSE, norm = FALSE
  )
  
  amp <- sp$amp
  # robust log-Skala
  S_db <- 20 * log10(amp + 1e-8)
  
  vals <- as.numeric(S_db)
  vals <- vals[is.finite(vals)]   # entferne NaN, Inf, -Inf
  
  # wenn zu wenig sinnvolle Werte übrig -> ungültiger Clip
  if (length(vals) < 10) {
    return(rep(NA_real_, 7L))
  }
  
  c(
    mean = mean(vals, na.rm = TRUE),
    sd   = sd(vals,   na.rm = TRUE),
    q10  = quantile(vals, 0.10, na.rm = TRUE),
    q25  = quantile(vals, 0.25, na.rm = TRUE),
    q50  = quantile(vals, 0.50, na.rm = TRUE),
    q75  = quantile(vals, 0.75, na.rm = TRUE),
    q90  = quantile(vals, 0.90, na.rm = TRUE)
  )
}

###############################################################################
# 3) One-Class SVM + PCA pro Machine-ID
###############################################################################

results_ocsvm_pca <- list()

for (id in unique_ids) {
  
  cat("\n==============================\n")
  cat("One-Class SVM + PCA – MACHINE ID:", id, "\n")
  cat("==============================\n")
  
  # Dateien pro ID
  train_idx <- which(train_ids == id)
  test_idx  <- which(test_ids  == id)
  
  train_files_id <- train_files[train_idx]
  test_files_id  <- test_files[test_idx]
  y_test_clip    <- y_test_all[test_idx]
  
  # Features berechnen
  cat(" -> Training Features…\n")
  train_feat_list <- pblapply(train_files_id, compute_poolstats)
  
  cat(" -> Test Features…\n")
  test_feat_list  <- pblapply(test_files_id, compute_poolstats)
  
  train_mat <- do.call(rbind, train_feat_list)
  test_mat  <- do.call(rbind, test_feat_list)
  
  # NA-Zeilen entfernen
  keep_train <- rowSums(is.na(train_mat)) == 0
  keep_test  <- rowSums(is.na(test_mat))  == 0
  
  train_mat <- train_mat[keep_train, , drop=FALSE]
  test_mat  <- test_mat[keep_test,  , drop=FALSE]
  y_test_clip_used <- y_test_clip[keep_test]
  
  if (nrow(train_mat) == 0 || nrow(test_mat) == 0) {
    cat("!! Keine gültigen Daten – ID", id, "übersprungen\n")
    next
  }
  
  ##########################################################################
  # 3a) PCA auf Trainingsdaten (normal) – dann Projektion für Train & Test
  ##########################################################################
  
  pca_obj <- prcomp(train_mat, center = TRUE, scale. = TRUE)
  
  # Anzahl PCs: min. Anzahl, die >= 95% Varianz erklären (mindestens 2)
  cumvar <- summary(pca_obj)$importance["Cumulative Proportion", ]
  k <- min(which(cumvar >= 0.95))
  if (is.infinite(k) || length(k) == 0) {
    k <- min(2L, ncol(train_mat))
  }
  k <- max(2L, k)  # mind. 2 Dimensionen
  
  cat(" -> PCA: benutze", k, "Hauptkomponenten\n")
  
  train_pca <- pca_obj$x[, 1:k, drop = FALSE]
  test_pca  <- predict(pca_obj, newdata = test_mat)[, 1:k, drop = FALSE]
  
  ##########################################################################
  # 3b) One-Class SVM im PCA-Raum trainieren
  ##########################################################################
  
  svm_model <- e1071::svm(
    x      = train_pca,
    y      = NULL,
    type   = "one-classification",
    kernel = "radial",
    nu     = 0.05,    # erlaubt ca. 5% Ausreißer im Training
    scale  = FALSE    # schon durch PCA zentriert/skaliert
  )
  
  ##########################################################################
  # 3c) Scores berechnen: -decision.values (höher = anomaler)
  ##########################################################################
  
  pred <- predict(svm_model, test_pca, decision.values = TRUE)
  dec  <- attr(pred, "decision.values")
  scores <- -as.numeric(dec)
  
  ##########################################################################
  # 3d) AUC / pAUC (DCASE)
  ##########################################################################
  
  auc_val <- pROC::auc(y_test_clip_used, scores)
  pauc_val <- pROC::auc(
    y_test_clip_used,
    scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  )
  
  cat("  AUC (PCA-OCSVM):  ", as.numeric(auc_val),  "\n")
  cat("  pAUC (PCA-OCSVM): ", as.numeric(pauc_val), "\n")
  
  results_ocsvm_pca[[id]] <- list(
    auc    = as.numeric(auc_val),
    pauc   = as.numeric(pauc_val),
    scores = scores,
    labels = y_test_clip_used,
    k_pca  = k
  )
}

###############################################################################
# 4) Zusammenfassung über alle IDs
###############################################################################

all_aucs_pca  <- sapply(results_ocsvm_pca, function(x) x$auc)
all_paucs_pca <- sapply(results_ocsvm_pca, function(x) x$pauc)

cat("\n===== ONE-CLASS SVM + PCA – SUMMARY =====\n")
print(all_aucs_pca)
#00        02        04        06 
#0.5371253 0.6342061 0.5616092 0.5731579 
cat("Mean AUC (PCA-OCSVM):  ", mean(all_aucs_pca),  "\n")
#Mean AUC (PCA-OCSVM):   0.5765246 
###############################################################################
