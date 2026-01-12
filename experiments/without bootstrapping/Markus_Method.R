###############################################################################
########  Quantile-Pooling-Statistik mit Quantildifferenzen (DCASE FAN)  #####
###############################################################################

setwd("C:/Users/ytid13aw/Documents/Resampling and Simulation Studies")

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

###########################
# 4) Pro-ID Training & Evaluation
###########################

results_qpool <- list()

for (id in unique_ids) {
  
  cat("\n==============================\n")
  cat("MACHINE ID (Pooling-Stats):", id, "\n")
  cat("==============================\n")
  
  # Indizes für diese ID
  train_idx <- which(train_ids == id)
  test_idx  <- which(test_ids  == id)
  
  train_files_id <- train_files[train_idx]              # nur normal
  test_files_id  <- test_files[test_idx]
  y_test_clip    <- y_test_clip_all[test_idx]           # Clip-Labels
  
  if (length(train_files_id) == 0 || length(test_files_id) == 0) {
    cat(">> Keine Dateien für ID", id, " – übersprungen.\n")
    next
  }
  
  # Pooling-Statistiken berechnen
  cat("  -> Features für Training (normal)…\n")
  train_feat_list <- pblapply(train_files_id, compute_poolstats)
  cat("  -> Features für Test…\n")
  test_feat_list  <- pblapply(test_files_id,  compute_poolstats)
  
  train_mat <- do.call(rbind, train_feat_list)
  test_mat  <- do.call(rbind, test_feat_list)
  
  # Zeilen mit NAs (z.B. defekte Files) raus
  keep_train <- rowSums(is.na(train_mat)) == 0
  keep_test  <- rowSums(is.na(test_mat))  == 0
  
  train_mat <- train_mat[keep_train, , drop = FALSE]
  test_mat  <- test_mat[keep_test,  , drop = FALSE]
  y_test_clip_used <- y_test_clip[keep_test]
  
  if (nrow(train_mat) == 0 || nrow(test_mat) == 0) {
    cat(">> Nach NA-Filter keine gültigen Features – ID", id, "übersprungen.\n")
    next
  }
  
  ###########################
  # 5) Quantil-Differenzen-Score
  ###########################
  # Idee: Für jede Pooling-Statistik (Spalte):
  #   – Referenz: Median (q50) und IQR der Trainings-Normaldaten
  #   – Score pro Clip = Mittel der |Feature - Median| / (IQR + eps)
  #   → je weiter ein Clip von der "normalen" Quantilverteilung weg ist,
  #     desto höher der Score.
  
  ref_median <- apply(train_mat, 2, median)
  ref_iqr    <- apply(train_mat, 2, IQR)
  eps        <- 1e-6
  
  score_fun <- function(x) {
    diffs <- abs(x - ref_median) / (ref_iqr + eps)
    mean(diffs)
  }
  
  scores <- apply(test_mat, 1, score_fun)
  
  ###########################
  # 6) AUC / pAUC (wie bei DCASE)
  ###########################
  
  auc_val <- pROC::auc(y_test_clip_used, scores)
  
  # pAUC im Bereich FPR <= 0.1
  # → entspricht Spezifität in [0.9, 1]
  pauc_val <- pROC::auc(
    y_test_clip_used,
    scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  )
  
  cat("  AUC:  ", as.numeric(auc_val),  "\n")
  cat("  pAUC: ", as.numeric(pauc_val), "\n")
  
  results_qpool[[id]] <- list(
    auc      = as.numeric(auc_val),
    pauc     = as.numeric(pauc_val),
    scores   = scores,
    labels   = y_test_clip_used,
    features = test_mat,
    ref_median = ref_median,
    ref_iqr    = ref_iqr
  )
}

###########################
# 7) Zusammenfassung
###########################

all_aucs_qpool  <- sapply(results_qpool, function(x) x$auc)
all_paucs_qpool <- sapply(results_qpool, function(x) x$pauc)

cat("\n==== Pooling-Quantil-Methode (global) ====\n")
print(all_aucs_qpool)
#00        02        04        06 
#0.5169042 0.6229248 0.6027874 0.5378393 
cat("Mean AUC:  ", mean(all_aucs_qpool),  "\n")
#Mean AUC:   0.5701139 
