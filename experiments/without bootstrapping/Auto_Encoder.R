###############################################################################
######################## DCASE FAN BASELINE (MLP-AE, Variante A) ##############
###############################################################################

setwd("C:/Users/ytid13aw/Documents/Resampling and Simulation Studies")

###########################
# 1) Python + TensorFlow
###########################

library(reticulate)
use_virtualenv("r-tf", required = TRUE)

library(tensorflow)
Sys.unsetenv("TF_USE_LEGACY_KERAS")

keras  <- tf$keras
layers <- keras$layers

###########################
# 2) R-Packages
###########################

library(pbapply)
library(pROC)
library(abind)

###########################
# 3) Parameter (alle als Integer, wo nötig)
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

epochs_ae     <- as.integer(100)
batch_size_ae <- as.integer(512)

###########################
# 4) Librosa
###########################

librosa <- import("librosa")
np      <- import("numpy")

###########################
# 5) Kontext-Features (640-dim) pro Clip
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
# 6) Dateien & IDs
###########################

train_files <- list.files(train_dir, pattern="\\.wav$", full.names=TRUE, recursive=TRUE)
test_files  <- list.files(test_dir,  pattern="\\.wav$", full.names=TRUE, recursive=TRUE)

cat("Train:", length(train_files), "\n")
cat("Test: ", length(test_files), "\n")

get_id <- function(x) gsub(".*id_(..)_.*", "\\1", basename(x))
train_ids  <- get_id(train_files)
test_ids   <- get_id(test_files)
unique_ids <- sort(unique(train_ids))

###########################
# 7) Modell (DCASE-MLP-AE)
###########################

build_model <- function() {
  inp <- keras$Input(shape = as.integer(input_dim))
  
  x <- layers$Dense(128L)(inp)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  # Bottleneck 8D
  x <- layers$Dense(8L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  # Decoder
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  x <- layers$Dense(128L)(x)
  x <- layers$BatchNormalization()(x)
  x <- layers$Activation("relu")(x)
  
  out <- layers$Dense(as.integer(input_dim))(x)
  
  model <- keras$Model(inp, out)
  
  model$compile(
    optimizer = keras$optimizers$Adam(learning_rate = 1e-3),
    loss      = "mse"
  )
  
  model
}

###########################
# 8) Pro-ID Training & Evaluation
###########################

results <- list()

for (id in unique_ids) {
  
  cat("\n==============================\n")
  cat("TRAINING MACHINE ID:", id, "\n")
  cat("==============================\n")
  
  # Dateien pro ID
  train_idx <- which(train_ids == id)
  test_idx  <- which(test_ids  == id)
  
  train_feat <- pblapply(train_files[train_idx], compute_logmel_context)
  test_feat  <- pblapply(test_files[test_idx],  compute_logmel_context)
  
  # NULL entfernen
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
  y_test_clip <- ifelse(grepl("anomaly", basename(test_files[test_idx]), ignore.case=TRUE), 1L, 0L)
  
  # Gruppenzugehörigkeit pro Frame (damit wir später zu Clips aggregieren können)
  frame_counts <- sapply(test_feat, nrow)
  test_groups  <- rep(seq_along(frame_counts), times = frame_counts)
  
  ###########################
  # Modell + Training
  ###########################
  
  model <- build_model()
  
  model$fit(
    X_train, X_train,
    epochs      = epochs_ae,          # als Integer definiert
    batch_size  = batch_size_ae,      # als Integer definiert
    shuffle     = TRUE,
    verbose     = 1L
  )
  
  ###########################
  # Rekonstruktion & Scores
  ###########################
  
  X_pred <- model$predict(X_test, batch_size = batch_size_ae)
  
  frame_scores <- rowMeans((X_test - X_pred)^2)
  
  # Clip-Level Score = Mittel über Frames pro Clip
  clip_scores <- tapply(frame_scores, test_groups, mean)
  
  ###########################
  # AUC / pAUC (DCASE)
  ###########################
  
  auc_val  <- pROC::auc(y_test_clip, clip_scores)
  pauc_val <- pROC::auc(
    y_test_clip,
    clip_scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  )
  
  cat("AUC:  ", as.numeric(auc_val),  "\n")
  cat("pAUC: ", as.numeric(pauc_val), "\n")
  
  results[[id]] <- list(
    auc    = as.numeric(auc_val),
    pauc   = as.numeric(pauc_val),
    scores = clip_scores,
    labels = y_test_clip
  )
}


all_aucs <- sapply(results, function(x) x$auc)
all_aucs
#00        02        04        06 
#0.5717445 0.7691643 0.6002586 0.8423269 

mean(all_aucs)
#[1] 0.6958736
###############################################################################