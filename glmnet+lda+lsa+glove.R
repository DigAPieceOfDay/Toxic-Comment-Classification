library(tidyverse)
library(tidytext)
library(magrittr)
library(doParallel)
library(text2vec)
library(tm)
library(glmnet)

# load other packages
library(pacman)
pacman::p_load_gh("trinker/gofastr")


# set core numbers
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "PSOCK")
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

#------------------------------- DATA CLEAN ------------------------------------
cat("Basic preprocessing & stats...\n")

# data imbalance process
# prop.table(table(train$toxic))
# 
# train[,targets] <- lapply(train[,targets],as.factor)
# train_balance <- SMOTE(
#   toxic ~ .,
#   train,
#   perc.over = 600,
#   perc.under = 100
# )


tr_te <- train %>% 
  select(-one_of(targets)) %>%
  bind_rows(test) %>%
  mutate(
    length = str_length(comment_text),
    use_cap = str_count(comment_text, "[A-Z]"),
    cap_len = use_cap / length,
    use_lower = str_count(comment_text, "[a-z]"),
    low_len = use_lower / length,
    cap_rate = ifelse(is.null(use_cap / use_lower), 0, use_cap / use_lower),
    cap_odds = ifelse(is.null(cap_len / low_len), 0, cap_len / low_len),
    use_exl = str_count(comment_text, fixed("!")),
    use_space = str_count(comment_text, fixed(" ")),
    use_double_space = str_count(comment_text, fixed("  ")),
    use_quest = str_count(comment_text, fixed("?")),
    use_punt = str_count(comment_text, "[[:punct:]]"),
    use_digit = str_count(comment_text, "[[:digit:]]"),
    digit_len = use_digit / length,
    use_break = str_count(comment_text, fixed("\n")),
    use_word = str_count(comment_text, "\\w+"),
    use_symbol = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^"),
    use_char = str_count(comment_text, "\\W*\\b\\w\\b\\W*"),
    use_i = str_count(comment_text, "(\\bI\\b)|(\\bi\\b)"),
    i_len = use_i / length,
    char_len = use_char / length,
    symbol_len = use_symbol / length,
    use_emotj = str_count(comment_text, "((?::|;|=)(?:-)?(?:\\)|D|P))"),
    cap_emo = use_emotj / length
  ) %>% 
  # select(-id) %T>% 
  glimpse()

# split train and test data
train_x <- tr_te[tri,]
test_x <- tr_te[-tri,]

# Remove all special chars, clean text and trasform words for comments
train_all_comments <- train_x %$%
  str_to_lower(comment_text) %>%
  
  # clear link
  str_replace_all("(f|ht)tp(s?)://\\S+", " ") %>%
  str_replace_all("http\\S+", "") %>%
  str_replace_all("xml\\S+", "") %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  
  # transform short forms
  str_replace_all("what's", "what is ") %>%
  str_replace_all("\\'s", " is ") %>%
  str_replace_all("\\'ve", " have ") %>%
  str_replace_all("can't", "cannot ") %>%
  str_replace_all("n't", " not ") %>%
  str_replace_all("i'm", "i am ") %>%
  str_replace_all("\\'re", " are ") %>%
  str_replace_all("\\'d", " would ") %>%
  str_replace_all("\\'ll", " will ") %>%
  str_replace_all("\\'scuse", " excuse ") %>%
  str_replace_all("pleas", " please ") %>%
  str_replace_all("sourc", " source ") %>%
  str_replace_all("peopl", " people ") %>%
  str_replace_all("remov", " remove ") %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  
  # transform shittext
  str_replace_all("(a|e)w+\\b", "") %>%
  str_replace_all("(y)a+\\b", "") %>%
  str_replace_all("(w)w+\\b", "") %>%
  str_replace_all("((a+)|(h+))(a+)((h+)?)\\b", "") %>%
  str_replace_all("((lol)(o?))+\\b", "") %>%
  str_replace_all("n ig ger", " nigger ") %>%
  str_replace_all("s hit", " shit ") %>%
  str_replace_all("g ay", " gay ") %>%
  str_replace_all("f ag got", " faggot ") %>%
  str_replace_all("c ock", " cock ") %>%
  str_replace_all("cu nt", " cunt ") %>%
  str_replace_all("idi ot", " idiot ") %>%
  str_replace_all("f u c k", " fuck ") %>%
  str_replace_all("fu ck", " fuck ") %>%
  str_replace_all("f u ck", " fuck ") %>%
  str_replace_all("c u n t", " cunt ") %>%
  str_replace_all("s u c k", " suck ") %>%
  str_replace_all("c o c k", " cock ") %>%
  str_replace_all("g a y", " gay ") %>%
  str_replace_all("ga y", " gay ") %>%
  str_replace_all("i d i o t", " idiot ") %>%
  str_replace_all("cocksu cking", "cock sucking") %>%
  str_replace_all("du mbfu ck", "dumbfuck") %>%
  str_replace_all("cu nt", "cunt") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "") %>%
  str_replace_all("(?<=\\w(ck))\\s(?=(ing)\\b)", "") %>%
  str_replace_all("(?<=\\b\\w)\\s(?=\\w\\b)", "") %>%
  str_replace_all("((lol)(o?))+", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "") %>%
  str_replace_all("(?<=\\w(uc))\\s(?=(ing)\\b)", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\w)", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(k)\\w)", "c") %>%
  
  # clean nicknames
  str_replace_all("@\\w+", " ") %>%
  
  # clean digit
  str_replace_all("[[:digit:]]", " ") %>%
  
  # remove linebreaks
  str_replace_all("\n", " ") %>%
  
  # remove graphics
  str_replace_all("[^[:graph:]]", " ") %>%
  
  # remove punctuation (if remain...)
  str_replace_all("[[:punct:]]", " ") %>%
  
  
  str_replace_all("[^[:alnum:]]", " ") %>%
  
  # remove single char
  str_replace_all("\\W*\\b\\w\\b\\W*", " ")  %>%
  
  # remove words with len < 2
  str_replace_all("\\b\\w{1,2}\\b", " ")  %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  str_replace_all("\\s+", " ") 


test_all_comments <- test_x %$%
  str_to_lower(comment_text) %>%
  
  # clear link
  str_replace_all("(f|ht)tp(s?)://\\S+", " ") %>%
  str_replace_all("http\\S+", "") %>%
  str_replace_all("xml\\S+", "") %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  
  # transform short forms
  str_replace_all("what's", "what is ") %>%
  str_replace_all("\\'s", " is ") %>%
  str_replace_all("\\'ve", " have ") %>%
  str_replace_all("can't", "cannot ") %>%
  str_replace_all("n't", " not ") %>%
  str_replace_all("i'm", "i am ") %>%
  str_replace_all("\\'re", " are ") %>%
  str_replace_all("\\'d", " would ") %>%
  str_replace_all("\\'ll", " will ") %>%
  str_replace_all("\\'scuse", " excuse ") %>%
  str_replace_all("pleas", " please ") %>%
  str_replace_all("sourc", " source ") %>%
  str_replace_all("peopl", " people ") %>%
  str_replace_all("remov", " remove ") %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  
  # transform shittext
  str_replace_all("(a|e)w+\\b", "") %>%
  str_replace_all("(y)a+\\b", "") %>%
  str_replace_all("(w)w+\\b", "") %>%
  str_replace_all("((a+)|(h+))(a+)((h+)?)\\b", "") %>%
  str_replace_all("((lol)(o?))+\\b", "") %>%
  str_replace_all("n ig ger", " nigger ") %>%
  str_replace_all("s hit", " shit ") %>%
  str_replace_all("g ay", " gay ") %>%
  str_replace_all("f ag got", " faggot ") %>%
  str_replace_all("c ock", " cock ") %>%
  str_replace_all("cu nt", " cunt ") %>%
  str_replace_all("idi ot", " idiot ") %>%
  str_replace_all("f u c k", " fuck ") %>%
  str_replace_all("fu ck", " fuck ") %>%
  str_replace_all("f u ck", " fuck ") %>%
  str_replace_all("c u n t", " cunt ") %>%
  str_replace_all("s u c k", " suck ") %>%
  str_replace_all("c o c k", " cock ") %>%
  str_replace_all("g a y", " gay ") %>%
  str_replace_all("ga y", " gay ") %>%
  str_replace_all("i d i o t", " idiot ") %>%
  str_replace_all("cocksu cking", "cock sucking") %>%
  str_replace_all("du mbfu ck", "dumbfuck") %>%
  str_replace_all("cu nt", "cunt") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "") %>%
  str_replace_all("(?<=\\w(ck))\\s(?=(ing)\\b)", "") %>%
  str_replace_all("(?<=\\b\\w)\\s(?=\\w\\b)", "") %>%
  str_replace_all("((lol)(o?))+", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "") %>%
  str_replace_all("(?<=\\w(uc))\\s(?=(ing)\\b)", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\w)", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(k)\\w)", "c") %>%
  
  # clean nicknames
  str_replace_all("@\\w+", " ") %>%
  
  # clean digit
  str_replace_all("[[:digit:]]", " ") %>%
  
  # remove linebreaks
  str_replace_all("\n", " ") %>%
  
  # remove graphics
  str_replace_all("[^[:graph:]]", " ") %>%
  
  # remove punctuation (if remain...)
  str_replace_all("[[:punct:]]", " ") %>%
  
  
  str_replace_all("[^[:alnum:]]", " ") %>%
  
  # remove single char
  str_replace_all("\\W*\\b\\w\\b\\W*", " ")  %>%
  
  # remove words with len < 2
  str_replace_all("\\b\\w{1,2}\\b", " ")  %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  str_replace_all("\\s+", " ") 



#-----------------------------WORD VECTORIZE------------------------------------
cat("Parsing comments...\n")

# 1.Vocabulary-based vectorization
it_train <- train_all_comments %>%
  itoken_parallel(
    tokenizer = word_tokenizer,
    ids = train_x$id,
    n_chunks = 4
  )

it_test <- test_all_comments %>%
  itoken_parallel(
    tokenizer = word_tokenizer,
    ids = test_x$id,
    n_chunks = 4
  )


# 2.create vocabulary by n-grams

## Generate Stopwords 
stops <- c(
  tm::stopwords("english"),
  tm::stopwords("SMART"),
  stopwords.custom
) %>%
  gofastr::prep_stopwords()

vocab <- create_vocabulary(
  it_train, 
  ngram = c(1L,2L),
  stopwords = stops
) %>%
prune_vocabulary(
  term_count_min = 5,
  doc_proportion_max = 0.5,
  doc_proportion_min = 0.001,
  vocab_term_max = 10000
)

# 3.vectorizer vocabulary
vectorizer <- vocab_vectorizer(vocab)

# 4.create dtm by tfidf and nomalize for l2
tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)

# create train dtm
dtm_train <- create_dtm(it_train, vectorizer) %>%
  fit_transform(tfidf)
dtm_train <- (dtm_train > 0) * 1

# create test dtm
dtm_test <- create_dtm(it_test, vectorizer) %>%
  fit_transform(tfidf)
dtm_test <- (dtm_test > 0) * 1

# --------------------------CREATE TOPIC MODEL---------------------------
# create lsa model
nlsa = 25
m_lsa <- LSA$new(n_topics = nlsa, method = "randomized")
lsa_train <- fit_transform(dtm_train, m_lsa)
lsa_test <- fit_transform(dtm_test, m_lsa)

colnames(lsa_train) <-
  sapply(1:ncol(lsa_train), function(x)
    paste("lsa1_", x, sep = ""))

colnames(lsa_test) <-
  sapply(1:ncol(lsa_test), function(x)
    paste("lsa1_", x, sep = ""))


# create lda model for train
nlda = 40
set.seed(233)
m_lda <- LDA$new(
  n_topics = nlda,
  doc_topic_prior = 50 / nlda,
  topic_word_prior =  1 / nlda
)

lda_train <- m_lda$fit_transform(
  dtm_train,
  n_iter = 500,
  convergence_tol = 0.001,
  check_convergence_every_n = 25
)
doc_topic_prob_train <- normalize(lda_train , norm = "l2")

# create lda model for test
lda_test <- m_lda$transform(dtm_test)
doc_topic_prob_test <- normalize(lda_test, norm = "l2")

# rename colname
colnames(lda_train) <-
  sapply(1:ncol(lda_train), function(x)
    paste("topic_distr_", x, sep = ""))
colnames(lda_test) <-
  sapply(1:ncol(lda_test), function(x)
    paste("topic_distr_", x, sep = ""))

colnames(doc_topic_prob_train) <-
  sapply(1:ncol(doc_topic_prob_train), function(x)
    paste("topic_prob_", x, sep = ""))
colnames(doc_topic_prob_test) <-
  sapply(1:ncol(doc_topic_prob_test), function(x)
    paste("topic_prob_", x, sep = ""))

#------------------------------GLOVE MODEL---------------------------------------
# use window of 5 for context words
tcm_train <- create_tcm(it_train, vectorizer, skip_grams_window = 5L)
tcm_test <- create_tcm(it_test, vectorizer, skip_grams_window = 5L)

glove <- GlobalVectors$new(
  word_vectors_size = 50,
  vocabulary = vocab,
  x_max = 10
)

# create word_vectors for train
word_vecs_train <- fit_transform(tcm_train, glove, n_iter = 100)
doc_vecs_train <- dtm_train %*% word_vecs_train

word_vecs_test <- fit_transform(tcm_test, glove, n_iter = 50)
doc_vecs_test <- dtm_test %*% word_vecs_test

#----------------------------------Cluster---------------------------------------------
cat("Preparing data for glmnet...\n")
Train_X <- train_x %>% 
  select(-c(comment_text,id)) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(lsa_train,doc_topic_prob_train,doc_vecs_train)

Test_X <- test_x %>% 
  select(-c(comment_text,id)) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(lsa_test,doc_topic_prob_test,doc_vecs_test)

rm(
  tr_te, 
  train, 
  test, 
  it_train, 
  it_test,
  vectorizer, 
  lsa_train, 
  lsa_test, 
  doc_vecs_train,
  doc_topic_prob_test,
  doc_vecs_train,
  doc_vecs_test
);
gc()


# cluster
dataset <- as.data.table(scale(rbind(Train_X, Test_X)))

set.seed(17)
cluster1 <- kmeans(dataset[, c(1:26)], centers = 7, iter.max = 15)

set.seed(17)
cluster2 <-
  kmeans(dataset[, c((26 + nlsa):ncol(dataset))], centers = 9, iter.max = 15)

set.seed(17)
cluster3 <- kmeans(dataset, centers = 17, iter.max = 15)




#----------------------------GLMNET TRAIN MODEL----------------------------------
options(scipen = 50)
cat("Training glmnet & predicting...\n")

for (target in targets) {
  cat("\nFitting", target, "...\n")
  Train_Y <- factor(train[[target]])
  m_glm <- cv.glmnet(
    Train_X,
    Train_Y,
    family = "binomial",
    type.measure = "auc",
    parallel = T,
    standardize = T,
    nfolds = 4,
    alpha = 0.001,
    nlambda = 100,
    thresh = 1e-3,
    maxit = 1e3
  )
  cat("\tAUC:", max(m_glm$cvm))
  submission[[target]] <- predict(m_glm,Test_X, type = "response", s = "lambda.min")
}

#----------------------------------SAVE DATA------------------------------------
cat("Creating submission file...\n")
write_csv(
  submission, 
  paste0("./submission/sample_submission_",Sys.Date(),".csv")
)

save.image(
  paste0("./output/toxic_classify_lsa_lda_glove", Sys.Date(), ".RData")
)







