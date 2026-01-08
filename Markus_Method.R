###############################################################################
########  Quantile-Pooling-Statistik mit Quantildifferenzen (DCASE FAN)  #####
###############################################################################

#setwd("C:/Users/ytid13aw/Documents/Resampling and Simulation Studies")

###########################
# 1) Pakete
###########################

library(tuneR)    # WAV einlesen
library(seewave)  # Spektrogramm
library(pbapply)  # Progress-Applikationen
library(pROC)     # AUC / pAUC

###########################
# 2) Daten / Pfade
###########################

train_dir <- "fan/train"
test_dir  <- "fan/test"

train_files <- list.files(train_dir, pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)
test_files  <- list.files(test_dir,  pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)

cat("Train:", length(train_files), "\n")
cat("Test: ",  length(test_files),  "\n")

# Machine-ID aus Dateinamen ziehen (wie beim AE)
get_id <- function(x) gsub(".*id_(..)_.*", "\\1", basename(x))

train_ids  <- get_id(train_files)
test_ids   <- get_id(test_files)
unique_ids <- sort(unique(train_ids))

# Clip-Level-Label (normal/anomaly) für Testdaten
y_test_clip_all <- ifelse(grepl("anomaly", basename(test_files), ignore.case = TRUE), 1L, 0L)

###########################
# 3) Spektrogramm + Pooling-Statistik pro Clip
###########################
# Wir nehmen ein STFT-Spektrogramm und berechnen globale Kennzahlen:
# mean, sd, q10, q25, q50, q75, q90 der (log-)Amplituden.

compute_poolstats <- function(file) {
  # WAV einlesen
  w <- tuneR::readWave(file)
  
  # Mono erzwingen
  if (w@stereo) {
    w <- mono(w, which = "left")
  }
  
  sr <- w@samp.rate
  # Spektrogramm (Amplitude)
  sp <- seewave::spectro(
    w,
    f    = sr,
    wl   = 1024,   # Fensterlänge
    ovlp = 50,     # 50% Overlap
    plot = FALSE,
    norm = FALSE
  )
  
  amp <- sp$amp  # Matrix: Frequenz x Zeit
  
  # In dB umrechnen, numerisch stabil
  S_db <- 20 * log10(amp + 1e-8)
  vals <- as.numeric(S_db)
  
  # Falls aus irgendeinem Grund leer → NA zurück
  if (length(vals) == 0 || all(!is.finite(vals))) {
    return(rep(NA_real_, 7L))
  }
  
  c(
    mean = mean(vals, na.rm = TRUE),
    sd   = sd(vals,   na.rm = TRUE),
    q10  = as.numeric(quantile(vals, 0.10, na.rm = TRUE)),
    q25  = as.numeric(quantile(vals, 0.25, na.rm = TRUE)),
    q50  = as.numeric(quantile(vals, 0.50, na.rm = TRUE)),
    q75  = as.numeric(quantile(vals, 0.75, na.rm = TRUE)),
    q90  = as.numeric(quantile(vals, 0.90, na.rm = TRUE))
  )
}

###############################################################################
# 4/7) Quantile-Pooling-Statistik MIT 100 Bootstrap-Runs (insgesamt)
#      -> Dieser Block ersetzt den bisherigen Abschnitt 4)–7) komplett.
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

y_test_clip_all_v <- y_test_clip_all[valid_test]

train_feat_all <- train_feat_all[valid_train]
test_feat_all  <- test_feat_all[valid_test]

unique_ids <- sort(unique(train_ids_v))

# --- Bootstrap settings (100 runs total) ---
R <- 100L
set.seed(123)

# Helper: one ID, one bootstrap sample
run_one_id_qpool <- function(id, boot_train_files_for_id) {
  
  # Training matrix (bootstrapped clips)
  train_feat_list <- train_feat_all[boot_train_files_for_id]
  train_feat_list <- train_feat_list[!sapply(train_feat_list, is.null)]
  if (length(train_feat_list) == 0) return(NULL)
  
  train_mat <- do.call(rbind, train_feat_list)
  if (is.null(train_mat) || nrow(train_mat) < 2) return(NULL)
  
  # Fixed test set for this ID
  test_idx <- which(test_ids_v == id)
  if (length(test_idx) == 0) return(NULL)
  
  test_files_id <- test_files_v[test_idx]
  test_feat_list <- test_feat_all[test_files_id]
  test_feat_list <- test_feat_list[!sapply(test_feat_list, is.null)]
  if (length(test_feat_list) == 0) return(NULL)
  
  test_mat <- do.call(rbind, test_feat_list)
  if (is.null(test_mat) || nrow(test_mat) < 2) return(NULL)
  
  y_test_clip_used <- y_test_clip_all_v[test_idx]
  
  # Reference from bootstrapped training normals
  ref_median <- apply(train_mat, 2, median)
  ref_iqr    <- apply(train_mat, 2, IQR)
  eps        <- 1e-6
  
  score_fun <- function(x) {
    diffs <- abs(x - ref_median) / (ref_iqr + eps)
    mean(diffs)
  }
  
  scores <- apply(test_mat, 1, score_fun)
  
  # Metrics
  auc_val <- as.numeric(pROC::auc(y_test_clip_used, scores))
  pauc_val <- as.numeric(pROC::auc(
    y_test_clip_used,
    scores,
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
    
    out <- run_one_id_qpool(id, boot_train_files_for_id)
    
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
saveRDS(results_df, file = "bootstrap_qpool_results.rds")
write.csv(results_df, file = "bootstrap_qpool_results.csv", row.names = FALSE)

###############################################################################
# Überblick
###############################################################################
# Mean/SD per ID across runs
summary_by_id <- aggregate(cbind(auc, pauc) ~ id, data = results_df,
                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
print(summary_by_id)

# Mean AUC per run across IDs
mean_auc_per_run <- aggregate(auc ~ run, data = results_df, FUN = mean)
cat("\nMean AUC across IDs per run:\n")
print(mean_auc_per_run)

cat("\nOverall mean AUC:", mean(results_df$auc), "\n")
cat("Overall mean pAUC:", mean(results_df$pauc), "\n")


cat("\n==== Pooling-Quantil-Methode (global) ====\n")
print(all_aucs_qpool)
#00        02        04        06 
#0.5169042 0.6229248 0.6027874 0.5378393 
cat("Mean AUC:  ", mean(all_aucs_qpool),  "\n")
#Mean AUC:   0.5701139 
