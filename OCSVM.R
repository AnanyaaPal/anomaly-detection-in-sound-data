###############################################################################
###################### One-Class SVM – DCASE FAN (pure R) #####################
###############################################################################

library(tuneR)
library(seewave)
library(pbapply)
library(pROC)
library(e1071)   # One-Class SVM

train_dir <- "fan/train"
test_dir  <- "fan/test"

train_files <- list.files(train_dir, pattern="\\.wav$", full.names=TRUE, recursive=TRUE)
test_files  <- list.files(test_dir,  pattern="\\.wav$", full.names=TRUE, recursive=TRUE)

get_id <- function(x) gsub(".*id_(..)_.*", "\\1", basename(x))
train_ids <- get_id(train_files)
test_ids  <- get_id(test_files)
unique_ids <- sort(unique(train_ids))

y_test_all <- ifelse(grepl("anomaly", basename(test_files), ignore.case=TRUE), 1L, 0L)

###############################################################################
# ROBUST: Pooling-Features pro Clip (7D)
###############################################################################

compute_poolstats <- function(file) {
  w <- tuneR::readWave(file)
  if (w@stereo) w <- mono(w, which = "left")
  
  sp <- seewave::spectro(
    w, f = w@samp.rate, wl = 1024, ovlp = 50,
    plot = FALSE, norm = FALSE
  )
  
  amp <- sp$amp
  # robust log-scale (verhindert -Inf)
  S_db <- 20 * log10(amp + 1e-8)
  
  vals <- as.numeric(S_db)
  vals <- vals[is.finite(vals)]   # entfernt NaN, Inf, -Inf
  
  # wenn nach Filterung zu wenige Daten vorhanden sind → ungültiger Clip
  if (length(vals) < 10) {
    return(rep(NA_real_, 7L))
  }
  
  c(
    mean = mean(vals, na.rm = TRUE),
    sd   = sd(vals, na.rm = TRUE),
    q10  = quantile(vals, 0.10, na.rm = TRUE),
    q25  = quantile(vals, 0.25, na.rm = TRUE),
    q50  = quantile(vals, 0.50, na.rm = TRUE),
    q75  = quantile(vals, 0.75, na.rm = TRUE),
    q90  = quantile(vals, 0.90, na.rm = TRUE)
  )
}

###############################################################################
# One-Class SVM pro Machine-ID
###############################################################################

results_ocsvm <- list()

for (id in unique_ids) {
  
  cat("\n==============================\n")
  cat("One-Class SVM – MACHINE ID:", id, "\n")
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
    cat("!! Keine gültigen Daten – ID übersprungen\n")
    next
  }
  
  ##########################################################################
  # One-Class SVM trainieren
  ##########################################################################
  
  svm_model <- e1071::svm(
    x = train_mat,
    y = NULL,
    type   = "one-classification",
    kernel = "radial",
    nu     = 0.05,     # erlaubt 5% Ausreißer im Training
    scale  = TRUE
  )
  
  ##########################################################################
  # Scores berechnen:
  # decision.values = Abstände zum Rand
  # Anomaly Score = -decision_value  (damit höher = anomal)
  ##########################################################################
  
  pred <- predict(svm_model, test_mat, decision.values = TRUE)
  dec  <- attr(pred, "decision.values")
  scores <- -as.numeric(dec)
  
  ##########################################################################
  # AUC / pAUC (DCASE-kompatibel)
  ##########################################################################
  
  auc_val <- pROC::auc(y_test_clip_used, scores)
  pauc_val <- pROC::auc(
    y_test_clip_used, scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  )
  
  cat("  AUC:  ", as.numeric(auc_val),  "\n")
  cat("  pAUC: ", as.numeric(pauc_val), "\n")
  
  results_ocsvm[[id]] <- list(
    auc    = as.numeric(auc_val),
    pauc   = as.numeric(pauc_val),
    scores = scores,
    labels = y_test_clip_used
  )
}

###############################################################################
# Zusammenfassung
###############################################################################

all_aucs  <- sapply(results_ocsvm, function(x) x$auc)
all_paucs <- sapply(results_ocsvm, function(x) x$pauc)

cat("\n===== ONE-CLASS SVM – SUMMARY =====\n")
print(all_aucs)
#00        02        04        06 
#0.5802457 0.6330641 0.6067816 0.6142936 
cat("Mean AUC:  ", mean(all_aucs), "\n")
#Mean AUC:   0.6085963 