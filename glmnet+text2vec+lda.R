library(tidyverse)
library(magrittr)
library(tm)
library(text2vec)
library(tokenizers)
library(glmnet)
library(doParallel)


# set core numbers
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores,type="PSOCK")
registerDoParallel(cl)

cat("Load data...\n")

train <- read_csv("./data/train.csv") 
test <- read_csv("./data/test.csv") 
submission <- read_csv("./data/sample_submission.csv") 

tri <- 1:nrow(train)
targets <- c(
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate"
)

#-------------------------------Basic FEATURE------------------------------------
cat("Basic preprocessing & stats...\n")
tr_te <- train %>% 
  select(-one_of(targets)) %>%
  bind_rows(test) %>%
  mutate(
    length = str_length(comment_text),
    ncap = str_count(comment_text, "[A-Z]"),
    ncap_len = ncap / length,
    nexcl = str_count(comment_text, fixed("!")),
    nquest = str_count(comment_text, fixed("?")),
    npunct = str_count(comment_text, "[[:punct:]]"),
    nword = str_count(comment_text, "\\w+"),
    nsymb = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^"),
    nsmile = str_count(comment_text, "((?::|;|=)(?:-)?(?:\\)|D|P))")
  ) %>%
# select(-id) %T>%
  glimpse()

# DATA CLEAN
system.time(
  comment_text <- parLapply(
    cl,
    tr_te$comment_text,
    cleaning_texts
  )
)



#-----------------------------FEATURE ENGINEER------------------------------------

cat("Parsing comments...\n")

# 1.Vocabulary-based vectorization
# it <- tr_te %$%
#   str_to_lower(comment_text) %>%
#   str_replace_all("[^[:alpha:]]", " ") %>%
#   str_replace_all("\\s+", " ") %>%
#   itoken(tokenizer = tokenize_word_stems,ids = tr_te$id)

it <- comment_text %>%
  itoken(tokenizer = tokenize_word_stems,ids = tr_te$id)


# 2.create vocabulary by n-grams
vocab <- create_vocabulary(
  it, 
  ngram = c(1,3),
  stopwords = stopwords("en")
) %>%
prune_vocabulary(
  term_count_min = 3,
  doc_proportion_max = 0.5,
  doc_proportion_min = 0.001,
  vocab_term_max = 4000
)

# 3.vectorizer vocabulary
vectorizer <- vocab_vectorizer(vocab)

# 4.create dtm by tfidf and nomalize for l2
tfidf <- TfIdf$new()
dtm <- create_dtm(it, vectorizer) %>%
  fit_transform(tfidf)

# create lsa model
m_lsa <- LSA$new(n_topics = 25, method = "randomized")
lsa <- fit_transform(dtm, m_lsa)

# create lda model
m_lda <- LDA$new(
  n_topics = 25,
  doc_topic_prior = 0.1,
  topic_word_prior = 0.01
)

lda <- m_lda$fit_transform(
  dtm,
  n_iter = 500,
  convergence_tol = 0.001,
  check_convergence_every_n = 25
)

lda$plot()

#-----------------------------------------------------------------------
# create glove model
# use window of 5 for context words
# tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
# 
# glove <- GlobalVectors$new(
#   word_vectors_size = 50,
#   vocabulary = vocab,
#   x_max = 10
# )
# wv_main <- fit_transform(tcm, glove, n_iter = 50)
# 
# wv_context <- glove$components
# dim(wv_context)
# 
# word_vectors <- wv_main + t(wv_context)


#-----------------------------------------------------------------------
cat("Preparing data for glmnet...\n")
X <- tr_te %>% 
  select(-c(comment_text,id)) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(dtm,lda)

X_train <- X[tri, ]
X_test <- X[-tri, ]


# rm(tr_te, test, tri, it, vectorizer, dtm, m_lda, lda); gc()
rm(it, vectorizer, dtm, m_lda, lda); gc()

#--------------------------------TRAIN MODEL-----------------------------------
cat("Training glmnet & predicting...\n")
for (target in targets) {
  cat("\nFitting", target, "...\n")
  y_train <- factor(train[[target]])
  m_glm <- cv.glmnet(
    X_train,
    y_train,
    alpha = 0,
    family = "binomial",
    type.measure = "auc",
    parallel = T,
    standardize = T,
    nfolds = 4,
    nlambda = 100,
    thresh = 1e-3,
    maxit = 1e3
  )
  cat("\tAUC:", max(m_glm$cvm))
  submission[[target]] <- predict(m_glm, X_test, type = "response", s = "lambda.min")
}

#----------------------------------SAVE DATA------------------------------------
cat("Creating submission file...\n")
write_csv(
  submission, 
  paste0("./submission/sample_submission_",Sys.Date(),".csv")
)

save.image("./output/toxic_classify.RData")


# ---------------------------Enaemble-----------------------------------
options(scipen = 50)

# Pool GRU + Fasttext -- 0.9830
submission_gru_fasttext <-
  read_csv("./ensemble/Pooled GRU_FastText.csv")

# lr with words and char n-grams -- 0.9788
submission_lr <-
  read_csv("./ensemble/LR with words and char n-grams.csv")

# Pool GRU + Glove -- 0.9824
submission_gru_glove <- read_csv("./ensemble/Pool GRU_Glove.csv")


# LDA + glmnet -- 0.9756
submission_lda_glmnet <- read_csv("./ensemble/Lda_glmnet.csv")


# Ensemble
w1 = 0.4
w2 = 0.3
w3 = 0.15
w4 = 0.15

toxic <-
  submission_gru_fasttext$toxic * w1 + submission_gru_glove$toxic * w2 + 
  submission_lr$toxic * w3 + submission_lda_glmnet$toxic * w4

severe_toxic <-  
  submission_gru_fasttext$severe_toxic * w1 + submission_gru_glove$severe_toxic * w2 + 
  submission_lr$severe_toxic * w3 + submission_lda_glmnet$severe_toxic * w4
  
obscene <- 
  submission_gru_fasttext$obscene * w1 + submission_gru_glove$obscene * w2 + 
  submission_lr$obscene * w3 + submission_lda_glmnet$obscene * w4

threat <- 
  submission_gru_fasttext$threat * w1 + submission_gru_glove$threat * w2 + 
  submission_lr$threat * w3 + submission_lda_glmnet$threat * w4

insult <- 
  submission_gru_fasttext$insult * w1 + submission_gru_glove$insult * w2 + 
  submission_lr$insult * w3 + submission_lda_glmnet$insult * w4

identity_hate <- 
  submission_gru_fasttext$identity_hate * w1 + submission_gru_glove$identity_hate * w2 + 
  submission_lr$identity_hate * w3 + submission_lda_glmnet$identity_hate * w4

  
simple_submission <- data.frame(
  id = subm$id,
  toxic = toxic,
  severe_toxic = severe_toxic,
  obscene = obscene,
  threat = threat,
  insult = insult,
  identity_hate = identity_hate
)

write_csv(
  simple_submission, 
  paste0("./submission/sample_submission_ensemble_2_",Sys.Date(),".csv")
)
