# Combining two GRU+CNN networks in order to increase prediction 
# acurancy([paper](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf)).


if (!require("keras")) install.packages("keras")
library(keras)
if (!require("tidyverse")) install.packages("tidyverse")
library(tidyverse)
if (!require("qdapRegex")) install.packages("qdapRegex")
library(qdapRegex)
if (!require("data.table")) install.packages("data.table")
library(data.table)


train_data <- read_csv("./data/train.csv")
test_data <- read_csv("./data/test.csv")

max_words = 13000  # 30000
maxl = 200

wordseq <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(c(train_data$comment_text, test_data$comment_text))

# word dictionary
word_index <- wordseq$word_index

x_train <- texts_to_sequences(wordseq, train_data$comment_text ) %>%
  pad_sequences(maxlen = maxl)
y_train <- as.matrix(train_data[,3:8])

x_test <- texts_to_sequences(wordseq, test_data$comment_text ) %>%
  pad_sequences(maxlen = maxl)

cat("beginning the word embedding \n")
############################################
#
#  WORD EMBEDDING
#
############################################

wgt <- fread("./data/glove.840B.300d.txt", data.table = FALSE)  %>%
  rename(word = V1)  %>%
  mutate(word = gsub("[[:punct:]]", " ", rm_white(word)))

dic_words <- wgt$word
wordindex <- unlist(wordseq$word_index)


dic <- data.frame(
  word = names(wordindex),
  key = wordindex,
  row.names = NULL
) %>%
  arrange(key) %>%
  .[1:max_words, ]

w_embed <- dic %>% left_join(wgt)

J = ncol(w_embed)
ndim = J-2
w_embed <- w_embed [1:(max_words-1),3:J] %>%
  mutate_all(as.numeric) %>%
  mutate_all(round,6) %>% 
  mutate_all(funs(replace(., is.na(.), 0)))    #fill na with 0

colnames(w_embed) = paste0("V",1:ndim)
w_embed <- rbind(rep(0, ndim), w_embed) %>%
  as.matrix()

#good weight format for the layer_embedding
w_embed <- list(array(w_embed , c(max_words, ndim)))

cat("beginning the neural network \n")
##################################################
#
#   NEURAL NETWORK MODEL 
#
##################################################

input <- layer_input(
  shape = list(maxl),
  dtype = "int32",
  name = "input"
)

filters = 64
kernel_size = 5

# create model1
model1 <- input %>%
  layer_embedding(
    input_dim = max_words,
    output_dim = ndim,
    input_length = maxl,
    weights = w_embed,
    trainable = FALSE
  ) %>%
  layer_spatial_dropout_1d(rate = 0.2) %>%
  bidirectional(layer_gru(units = 40, return_sequences = TRUE, recurrent_dropout = 0.1)) %>%
  layer_conv_1d(
    60, 
    3, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) 

model2 <- input %>%
  layer_embedding(
    input_dim = max_words,
    output_dim = ndim,
    input_length = maxl,
    weights = w_embed,
    trainable = FALSE
  ) %>%
  layer_spatial_dropout_1d(rate = 0.1) %>%
  bidirectional(
    layer_gru(units = 80, return_sequences = TRUE, recurrent_dropout = 0.1) 
  ) %>% 
  layer_conv_1d(
    120, 
    2, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) 


max_pool1 <- model1 %>% layer_global_max_pooling_1d()
ave_pool1 <- model1 %>% layer_global_average_pooling_1d()
max_pool2 <- model2 %>% layer_global_max_pooling_1d()
ave_pool2 <- model2 %>% layer_global_average_pooling_1d()


output <- layer_concatenate(list(ave_pool1, max_pool1, ave_pool2, max_pool2)) %>%
  layer_dense(units = 6, activation = "sigmoid")

model <- keras_model(input, output)

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 3,
  batch_size = 32,
  validation_split = 0.05,
  callbacks = list(
    callback_model_checkpoint(
      paste0(
        "./output/toxic_comment_model.h5"
      ),
      save_best_only = TRUE
    ),
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 0,
      patience = 0,
      verbose = 0,
      mode = c("auto", "min", "max")
    )
  )
)

# load model
# model = load_model_hdf5(
#   paste0(
#     "./model/toxic_comment_model.h5"
#   )
# )

cat("beginning the prediction & submission \n")
###########################################
#
# PREDICTION & SUBMISSON
#
###########################################

pred <- model %>%
  predict(x_test, batch_size = 1024) %>%
  as.data.frame()

pred <- cbind(id = test_data$id, pred) 

names(pred)[2:7] <- c(
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate"
)

write_csv(pred,"./submission/submission_gru_max_avg_tune.csv")