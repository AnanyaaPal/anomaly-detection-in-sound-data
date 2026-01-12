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
# One-Class SVM MIT 100 Bootstrap-Runs (insgesamt) + AUCs speichern
# -> Dieser Block ersetzt den bisherigen "One-Class SVM pro Machine-ID" Loop
#    UND die anschließende Zusammenfassung komplett.
###############################################################################

# --- Feature cache: compute once per file (sonst viel zu langsam) ---
cat("\nPrecompute poolstats train features (once)...\n")
train_feat_all <- pblapply(train_files, compute_poolstats)
names(train_feat_all) <- train_files

cat("\nPrecompute poolstats test features (once)...\n")
test_feat_all <- pblapply(test_files, compute_poolstats)
names(test_feat_all) <- test_files

# Remove invalid (NA) rows globally
is_valid_vec <- function(v) is.numeric(v) && length(v) == 7L && all(is.finite(v))

valid_train <- sapply(train_feat_all, is_valid_vec)
valid_test  <- sapply(test_feat_all,  is_valid_vec)

train_files_v <- train_files[valid_train]
test_files_v  <- test_files[valid_test]

train_ids_v <- train_ids[valid_train]
test_ids_v  <- test_ids[valid_test]

y_test_all_v <- y_test_all[valid_test]

train_feat_all <- train_feat_all[valid_train]
test_feat_all  <- test_feat_all[valid_test]

unique_ids <- sort(unique(train_ids_v))

# --- Bootstrap settings (100 runs total) ---
R <- 100L
set.seed(123)

# Helper: one ID, one bootstrap sample
run_one_id_ocsvm <- function(id, boot_train_files_for_id) {
  
  # Training matrix (bootstrapped clips)
  train_feat_list <- train_feat_all[boot_train_files_for_id]
  train_feat_list <- train_feat_list[!sapply(train_feat_list, is.null)]
  if (length(train_feat_list) == 0) return(NULL)
  
  train_mat <- do.call(rbind, train_feat_list)
  if (is.null(train_mat) || nrow(train_mat) < 5) return(NULL)
  
  # Fixed test set for this ID
  test_idx <- which(test_ids_v == id)
  if (length(test_idx) == 0) return(NULL)
  
  test_files_id <- test_files_v[test_idx]
  test_feat_list <- test_feat_all[test_files_id]
  test_feat_list <- test_feat_list[!sapply(test_feat_list, is.null)]
  if (length(test_feat_list) == 0) return(NULL)
  
  test_mat <- do.call(rbind, test_feat_list)
  if (is.null(test_mat) || nrow(test_mat) < 2) return(NULL)
  
  y_test_clip_used <- y_test_all_v[test_idx]
  
  ##########################################################################
  # One-Class SVM trainieren
  ##########################################################################
  svm_model <- e1071::svm(
    x = train_mat,
    y = NULL,
    type   = "one-classification",
    kernel = "radial",
    nu     = 0.05,
    scale  = TRUE
  )
  
  ##########################################################################
  # Scores: Anomaly Score = -decision_value (höher = anomal)
  ##########################################################################
  pred <- predict(svm_model, test_mat, decision.values = TRUE)
  dec  <- attr(pred, "decision.values")
  scores <- -as.numeric(dec)
  
  ##########################################################################
  # AUC / pAUC (DCASE-kompatibel)
  ##########################################################################
  auc_val <- as.numeric(pROC::auc(y_test_clip_used, scores))
  pauc_val <- as.numeric(pROC::auc(
    y_test_clip_used, scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  ))
  
  list(auc = auc_val, pauc = pauc_val)
}

###############################################################################
# Main bootstrap loop (100 runs total) + store all AUCs
###############################################################################
results_df <- data.frame(
  run  = integer(0),
  id   = character(0),
  auc  = numeric(0),
  pauc = numeric(0),
  stringsAsFactors = FALSE
)

pb <- txtProgressBar(min = 0, max = R * length(unique_ids), style = 3)
kk <- 0L

for (r in seq_len(R)) {
  for (id in unique_ids) {
    
    train_idx <- which(train_ids_v == id)
    if (length(train_idx) == 0) { kk <- kk + 1L; setTxtProgressBar(pb, kk); next }
    
    train_files_id <- train_files_v[train_idx]
    
    # Bootstrap resampling of training CLIPS (same size, with replacement)
    boot_train_files_for_id <- sample(
      train_files_id,
      size    = length(train_files_id),
      replace = TRUE
    )
    
    out <- run_one_id_ocsvm(id, boot_train_files_for_id)
    
    if (!is.null(out)) {
      results_df <- rbind(
        results_df,
        data.frame(run = r, id = id, auc = out$auc, pauc = out$pauc, stringsAsFactors = FALSE)
      )
    }
    
    kk <- kk + 1L
    setTxtProgressBar(pb, kk)
  }
}
close(pb)

# Save raw results (one row per run+id)
saveRDS(results_df, file = "bootstrap_ocsvm_results.rds")
write.csv(results_df, file = "bootstrap_ocsvm_results.csv", row.names = FALSE)

###############################################################################
# Überblick
###############################################################################
summary_by_id <- aggregate(cbind(auc, pauc) ~ id, data = results_df,
                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
print(summary_by_id)

mean_auc_per_run <- aggregate(auc ~ run, data = results_df, FUN = mean)
cat("\nMean AUC across IDs per run:\n")
print(mean_auc_per_run)

cat("\nOverall mean AUC:", mean(results_df$auc), "\n")
cat("Overall mean pAUC:", mean(results_df$pauc), "\n")


cat("\n===== ONE-CLASS SVM – SUMMARY =====\n")
print(all_aucs)
#00        02        04        06 
#0.5802457 0.6330641 0.6067816 0.6142936 
cat("Mean AUC:  ", mean(all_aucs), "\n")
#Mean AUC:   0.6085963 