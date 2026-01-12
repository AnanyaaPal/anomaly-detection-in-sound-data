###############################################################################
###################### DCASE FAN – IF (R-only, librosa-like) ##################
###############################################################################

#setwd("C:/Users/ytid13aw/Documents/Resampling and Simulation Studies")

###########################
# 1) Packages
###########################
library(tuneR)    # WAV lesen
library(seewave)  # resamp
library(pbapply)
library(pROC)
library(isotree)

###########################
# 2) Parameter
###########################
sample_rate <- as.integer(16000)
n_fft       <- as.integer(1024)
hop_length  <- as.integer(512)
n_mels      <- as.integer(128)

P         <- as.integer(2)
n_context <- as.integer(2L * P + 1L)
input_dim <- as.integer(n_mels * n_context)

train_dir <- "fan/train"
test_dir  <- "fan/test"

###############################################################################
# 3) librosa-like helpers
###############################################################################

# Hann window (periodic=True wie librosa default)
hann_window <- function(n) {
  # periodic Hann: win[n] = 0.5 - 0.5*cos(2*pi*k/n), k=0..n-1
  k <- 0:(n - 1)
  0.5 - 0.5 * cos(2 * pi * k / n)
}

# librosa center padding: pad both sides by n_fft/2
pad_center <- function(y, n_fft) {
  pad <- as.integer(floor(n_fft / 2))
  c(rep(0, pad), y, rep(0, pad))
}

# Frame slicing like librosa.util.frame (no partial frame at end)
frame_signal <- function(y, frame_length, hop_length) {
  n <- length(y)
  if (n < frame_length) return(matrix(numeric(0), nrow = frame_length, ncol = 0))
  n_frames <- 1L + as.integer(floor((n - frame_length) / hop_length))
  idx0 <- seq.int(0L, by = hop_length, length.out = n_frames)
  out <- matrix(0, nrow = frame_length, ncol = n_frames)
  for (i in seq_len(n_frames)) {
    start <- idx0[i] + 1L
    out[, i] <- y[start:(start + frame_length - 1L)]
  }
  out
}

# STFT returning complex matrix (n_fft/2+1 x T), librosa-like
stft_librosa_like <- function(y, n_fft, hop_length, center = TRUE) {
  if (center) y <- pad_center(y, n_fft)
  
  frames <- frame_signal(y, frame_length = n_fft, hop_length = hop_length)
  if (ncol(frames) == 0) return(matrix(complex(real = numeric(0), imaginary = numeric(0)),
                                       nrow = floor(n_fft/2) + 1L, ncol = 0))
  
  win <- hann_window(n_fft)
  frames <- frames * win
  
  n_freqs <- as.integer(floor(n_fft / 2) + 1L)
  S <- matrix(complex(real = 0, imaginary = 0), nrow = n_freqs, ncol = ncol(frames))
  
  for (t in seq_len(ncol(frames))) {
    X <- fft(frames[, t])
    S[, t] <- X[1:n_freqs]
  }
  S
}

# Slaney mel conversions (librosa uses this by default)
hz_to_mel_slaney <- function(f) {
  # piecewise: linear below 1000 Hz, log above
  f <- as.numeric(f)
  f_min <- 0
  f_sp <- 200.0 / 3
  mels <- (f - f_min) / f_sp
  
  min_log_hz <- 1000.0
  min_log_mel <- (min_log_hz - f_min) / f_sp
  logstep <- log(6.4) / 27.0
  
  idx <- f >= min_log_hz
  mels[idx] <- min_log_mel + log(f[idx] / min_log_hz) / logstep
  mels
}

mel_to_hz_slaney <- function(m) {
  m <- as.numeric(m)
  f_min <- 0
  f_sp <- 200.0 / 3
  f <- f_min + f_sp * m
  
  min_log_hz <- 1000.0
  min_log_mel <- (min_log_hz - f_min) / f_sp
  logstep <- log(6.4) / 27.0
  
  idx <- m >= min_log_mel
  f[idx] <- min_log_hz * exp(logstep * (m[idx] - min_log_mel))
  f
}

# librosa-like mel filterbank: Slaney + norm="slaney"
mel_filterbank_librosa_like <- function(sr, n_fft, n_mels, fmin = 0, fmax = sr/2) {
  n_freqs <- as.integer(floor(n_fft/2) + 1L)
  fftfreqs <- seq(0, sr/2, length.out = n_freqs)
  
  # mel points
  m_min <- hz_to_mel_slaney(fmin)
  m_max <- hz_to_mel_slaney(fmax)
  m_pts <- seq(m_min, m_max, length.out = n_mels + 2L)
  f_pts <- mel_to_hz_slaney(m_pts)
  
  # bins
  bins <- floor((n_fft + 1) * f_pts / sr)
  bins[bins < 0] <- 0
  bins[bins > (n_freqs - 1)] <- n_freqs - 1
  
  fb <- matrix(0, nrow = n_mels, ncol = n_freqs)
  
  for (m in seq_len(n_mels)) {
    left   <- bins[m]     + 1L
    center <- bins[m + 1] + 1L
    right  <- bins[m + 2] + 1L
    
    if (center <= left) center <- min(left + 1L, n_freqs)
    if (right <= center) right <- min(center + 1L, n_freqs)
    
    if (center > left) {
      fb[m, left:center] <- (seq(left, center) - left) / (center - left)
    }
    if (right > center) {
      fb[m, center:right] <- (right - seq(center, right)) / (right - center)
    }
  }
  
  # Slaney normalization: scale each filter by 2/(f_{m+2}-f_m)
  # (area normalization to make approximately constant energy per band)
  enorm <- 2.0 / (f_pts[3:(n_mels + 2L)] - f_pts[1:n_mels])
  fb <- fb * matrix(enorm, nrow = n_mels, ncol = n_freqs)
  
  fb
}

# librosa.power_to_db equivalent (for power)
power_to_db_librosa_like <- function(S, ref = NULL, amin = 1e-10, top_db = 80) {
  S <- pmax(S, amin)
  if (is.null(ref)) ref <- max(S)
  log_spec <- 10 * log10(S)
  log_spec <- log_spec - 10 * log10(pmax(amin, ref))
  if (!is.null(top_db)) {
    log_spec <- pmax(log_spec, max(log_spec) - top_db)
  }
  log_spec
}

###############################################################################
# 4) Precompute Mel FB (fixed dims now: n_fft -> 513 bins always)
###############################################################################
fb_global <- mel_filterbank_librosa_like(sample_rate, n_fft, n_mels)

###############################################################################
# 5) Feature extraction: log-mel + context (librosa-like)
###############################################################################
compute_logmel_context <- function(file) {
  wav <- tuneR::readWave(file)
  
  # mono
  if (wav@stereo) y <- (wav@left + wav@right) / 2 else y <- wav@left
  y <- as.numeric(y)
  
  # normalize to [-1,1] robustly (librosa load returns float typically)
  m <- max(abs(y))
  if (is.finite(m) && m > 0) y <- y / m
  
  # resample if needed
  sr_in <- wav@samp.rate
  if (sr_in != sample_rate) {
    y <- seewave::resamp(y, f = sr_in, g = sample_rate, output = "vector")
  }
  
  # STFT (center=TRUE like librosa default)
  S_c <- stft_librosa_like(y, n_fft = n_fft, hop_length = hop_length, center = TRUE)
  
  # Power spectrogram
  S_pow <- (Mod(S_c))^2  # 513 x T
  
  # Mel power
  mel_power <- fb_global %*% S_pow  # 128 x T
  
  # dB
  M <- power_to_db_librosa_like(mel_power, ref = max(mel_power), amin = 1e-10, top_db = 80)
  
  n_frames <- ncol(M)
  if (n_frames < n_context) return(NULL)
  
  T_eff <- n_frames - 2L * P
  out   <- matrix(0, nrow = T_eff, ncol = input_dim)
  
  idx <- 1L
  for (t in seq.int(P + 1L, n_frames - P)) {
    win <- M[, (t - P):(t + P), drop = FALSE]   # 128 x 5
    out[idx, ] <- as.numeric(win)               # 640
    idx <- idx + 1L
  }
  out
}

###############################################################################
# 6) Files & IDs
###############################################################################
train_files <- list.files(train_dir, pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)
test_files  <- list.files(test_dir,  pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)

cat("Train:", length(train_files), "\n")
cat("Test: ", length(test_files),  "\n")

get_id <- function(x) gsub(".*id_(..)_.*", "\\1", basename(x))
train_ids  <- get_id(train_files)
test_ids   <- get_id(test_files)
unique_ids <- sort(unique(train_ids))

###############################################################################
# 7) Precompute Features ONCE (cache), then 100 Bootstrap runs total
###############################################################################

# --- Feature cache: compute once per file ---
cat("\nPrecompute train features (once)...\n")
train_feat_all <- pblapply(train_files, compute_logmel_context)
names(train_feat_all) <- train_files

cat("\nPrecompute test features (once)...\n")
test_feat_all <- pblapply(test_files, compute_logmel_context)
names(test_feat_all) <- test_files

# Drop NULL files globally (too short etc.)
valid_train <- !sapply(train_feat_all, is.null)
valid_test  <- !sapply(test_feat_all,  is.null)

train_files_v <- train_files[valid_train]
test_files_v  <- test_files[valid_test]

train_ids_v <- train_ids[valid_train]
test_ids_v  <- test_ids[valid_test]

train_feat_all <- train_feat_all[valid_train]
test_feat_all  <- test_feat_all[valid_test]

unique_ids <- sort(unique(train_ids_v))

###############################################################################
# 8) Bootstrap settings (100 runs total)
###############################################################################
R <- 100L
set.seed(123)

###############################################################################
# 9) Helper: train+eval for one ID given bootstrapped train file list
###############################################################################
run_one_id_if <- function(id, boot_train_files_for_id,
                          max_frames_per_clip = NULL) {
  
  # Training features (list of matrices)
  train_feat_list <- train_feat_all[boot_train_files_for_id]
  if (length(train_feat_list) == 0) return(NULL)
  
  # Optional speed-up: cap frames per train clip
  if (!is.null(max_frames_per_clip)) {
    max_frames_per_clip <- as.integer(max_frames_per_clip)
    train_feat_list <- lapply(train_feat_list, function(M) {
      if (is.null(M) || nrow(M) == 0) return(NULL)
      if (nrow(M) <= max_frames_per_clip) return(M)
      M[sample.int(nrow(M), max_frames_per_clip), , drop = FALSE]
    })
    train_feat_list <- train_feat_list[!sapply(train_feat_list, is.null)]
    if (length(train_feat_list) == 0) return(NULL)
  }
  
  X_train <- do.call(rbind, train_feat_list)
  if (is.null(X_train) || nrow(X_train) < 2) return(NULL)
  
  # Fixed test set for this ID
  test_idx <- which(test_ids_v == id)
  if (length(test_idx) == 0) return(NULL)
  
  test_files_id <- test_files_v[test_idx]
  test_feat_list <- test_feat_all[test_files_id]
  test_feat_list <- test_feat_list[!sapply(test_feat_list, is.null)]
  if (length(test_feat_list) == 0) return(NULL)
  
  X_test <- do.call(rbind, test_feat_list)
  if (is.null(X_test) || nrow(X_test) < 2) return(NULL)
  
  # Clip labels from filename
  y_test_clip <- ifelse(
    grepl("anomaly", basename(test_files_id), ignore.case = TRUE),
    1L, 0L
  )
  
  # Frame -> clip grouping for aggregation
  frame_counts <- sapply(test_feat_list, nrow)
  test_groups  <- rep(seq_along(frame_counts), times = frame_counts)
  
  # Train Isolation Forest
  ss <- min(256L, nrow(X_train))
  iso_model <- isolation.forest(
    X_train,
    ntrees      = as.integer(200),
    sample_size = as.integer(ss),
    nthreads    = as.integer(1)
  )
  
  # Predict + aggregate to clip score
  frame_scores <- as.numeric(predict(iso_model, X_test))
  clip_scores  <- as.numeric(tapply(frame_scores, test_groups, mean))
  
  # Metrics
  auc_val <- as.numeric(pROC::auc(y_test_clip, clip_scores))
  pauc_val <- as.numeric(pROC::auc(
    y_test_clip,
    clip_scores,
    partial.auc       = c(1, 0.9),
    partial.auc.focus = "specificity"
  ))
  
  list(auc = auc_val, pauc = pauc_val)
}

###############################################################################
# 10) Main bootstrap loop (100 runs total) + save all AUCs
###############################################################################
results_df <- data.frame(
  run  = integer(0),
  id   = character(0),
  auc  = numeric(0),
  pauc = numeric(0),
  stringsAsFactors = FALSE
)

pb <- txtProgressBar(min = 0, max = R * length(unique_ids), style = 3)
k <- 0L

for (r in seq_len(R)) {
  for (id in unique_ids) {
    
    train_idx <- which(train_ids_v == id)
    if (length(train_idx) == 0) { k <- k + 1L; setTxtProgressBar(pb, k); next }
    
    train_files_id <- train_files_v[train_idx]
    
    # Bootstrap resampling of training CLIPS (same size, with replacement)
    boot_train_files_for_id <- sample(
      train_files_id,
      size    = length(train_files_id),
      replace = TRUE
    )
    
    # Set max_frames_per_clip to a number (e.g. 200L) to speed up, or leave NULL
    out <- run_one_id_if(id, boot_train_files_for_id, max_frames_per_clip = NULL)
    
    if (!is.null(out)) {
      results_df <- rbind(
        results_df,
        data.frame(run = r, id = id, auc = out$auc, pauc = out$pauc, stringsAsFactors = FALSE)
      )
    }
    
    k <- k + 1L
    setTxtProgressBar(pb, k)
  }
}
close(pb)

# Save raw results (one row per run+id)
saveRDS(results_df, file = "bootstrap_if_results.rds")
write.csv(results_df, file = "bootstrap_if_results.csv", row.names = FALSE)

###############################################################################
# 11) Quick overview
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



###############################################################################

###############################################################################

#       00        02        04        06 
#0.5371253 0.6463788 0.5154885 0.6197230 
mean(all_aucs_if)
#[1] 0.5796789