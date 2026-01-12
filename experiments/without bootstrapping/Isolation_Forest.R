###############################################################################
###################### DCASE FAN – ISOLATION FOREST (R) ######################
###############################################################################

setwd("C:/Users/ytid13aw/Documents/Resampling and Simulation Studies")

###########################
# 1) Python (nur für Features)
###########################

library(reticulate)
use_virtualenv("r-tf", required = TRUE)

librosa <- import("librosa")
np      <- import("numpy")

###########################
# 2) R-Packages
###########################

library(pbapply)
library(pROC)
library(isotree)   # <— Isolation Forest in R

###########################
# 3) Parameter (Integer wo nötig)
###########################

sample_rate <- as.integer(16000)
n_fft       <- as.integer(1024)
hop_length  <- as.integer(512)
n_mels      <- as.integer(128)

P         <- as.integer(2)                       # Kontext links/rechts
n_context <- as.integer(2L * P + 1L)             # 5 Frames
input_dim <- as.integer(n_mels * n_context)      # 128 * 5 = 640

train_dir <- "fan/train"
test_dir  <- "fan/test"

###########################
# 4) Kontext-Features (640-dim) pro Clip
###########################

compute_logmel_context <- function(file) {
  # Wave laden
  y_sr <- librosa$load(file, sr = sample_rate, mono = TRUE)
  y    <- y_sr[[1]]
  
  # Mel-Spectrogram
  mel <- librosa$feature$melspectrogram(
    y          = y,
    sr         = sample_rate,
    n_fft      = n_fft,
    hop_length = hop_length,
    n_mels     = n_mels
  )
  
  mel_db <- librosa$power_to_db(mel, ref = np$max)
  M      <- mel_db[]
  
  # Sicherstellen, dass M Matrix ist (n_mels x T)
  if (!is.matrix(M)) {
    M <- matrix(M, nrow = n_mels)
  }
  
  n_frames <- ncol(M)
  
  # zu kurze Dateien verwerfen
  if (n_frames < n_context) {
    return(NULL)
  }
  
  T_eff <- n_frames - 2L * P
  out   <- matrix(0, nrow = T_eff, ncol = input_dim)
  
  idx <- 1L
  for (t in seq.int(P + 1L, n_frames - P)) {
    win <- M[, (t - P):(t + P), drop = FALSE]  # (128 x 5)
    out[idx, ] <- as.numeric(win)             # 640-dim
    idx <- idx + 1L
  }
  
  out
}

###########################
# 5) Dateien & Machine IDs
###########################

train_files <- list.files(train_dir, pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)
test_files  <- list.files(test_dir,  pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)

cat("Train:", length(train_files), "\n")
cat("Test: ", length(test_files),  "\n")

get_id <- function(x) gsub(".*id_(..)_.*", "\\1", basename(x))

train_ids  <- get_id(train_files)
test_ids   <- get_id(test_files)
unique_ids <- sort(unique(train_ids))

###########################
# 6) Pro-ID Training & Evaluation (Isolation Forest)
###########################

results_if <- list()

for (id in unique_ids) {
  
  cat("\n==============================\n")
  cat("ISOLATION FOREST – MACHINE ID:", id, "\n")
  cat("==============================\n")
  
  # Dateien pro ID
  train_idx <- which(train_ids == id)
  test_idx  <- which(test_ids  == id)
  
  train_feat <- pblapply(train_files[train_idx], compute_logmel_context)
  test_feat  <- pblapply(test_files[test_idx],  compute_logmel_context)
  
  # NULL entfernen (zu kurze Clips etc.)
  train_feat <- train_feat[!sapply(train_feat, is.null)]
  test_feat  <- test_feat[!sapply(test_feat,  is.null)]
  
  if (length(train_feat) == 0 || length(test_feat) == 0) {
    cat(">> Keine gültigen Features für ID", id, " – wird übersprungen.\n")
    next
  }
  
  # Stack (Frames × 640)
  X_train <- do.call(rbind, train_feat)
  X_test  <- do.call(rbind, test_feat)
  
  # Clip-Labels (normal/anomaly auf Clip-Ebene)
  y_test_clip <- ifelse(
    grepl("anomaly", basename(test_files[test_idx]), ignore.case = TRUE),
    1L, 0L
  )
  
  # Gruppenzugehörigkeit pro Frame → Clip-Level
  frame_counts <- sapply(test_feat, nrow)
  test_groups  <- rep(seq_along(frame_counts), times = frame_counts)
  
  ###########################
  # Isolation Forest Modell
  ###########################
  
  iso_model <- isolation.forest(
    X_train,
    ntrees      = as.integer(200),
    sample_size = as.integer(256),
    nthreads    = as.integer(1)
  )
  
  ###########################
  # Scores (Frame → Clip)
  ###########################
  
  # Frame-Level Outlier Scores (höher = "anomaler")
  frame_scores <- as.numeric(predict(iso_model, X_test))
  
  # Clip-Level Score = Mittel über Frames pro Clip
  clip_scores <- tapply(frame_scores, test_groups, mean)
  
  ###########################
  # AUC / pAUC
  ###########################
  
  auc_val  <- pROC::auc(y_test_clip, clip_scores)
  pauc_val <- pROC::auc(
    y_test_clip,
    clip_scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  )
  
  cat("AUC (IF):  ", as.numeric(auc_val),  "\n")
  cat("pAUC (IF): ", as.numeric(pauc_val), "\n")
  
  results_if[[id]] <- list(
    auc    = as.numeric(auc_val),
    pauc   = as.numeric(pauc_val),
    scores = clip_scores,
    labels = y_test_clip
  )
}

###########################
# 7) Überblick AUCs
###########################

all_aucs_if <- sapply(results_if, function(x) x$auc)
all_aucs_if
mean(all_aucs_if)
###############################################################################

#       00        02        04        06 
#0.5371253 0.6463788 0.5154885 0.6197230 
mean(all_aucs_if)
#[1] 0.5796789