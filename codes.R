library(jsonlite)
library(dplyr)
library(tm)
library(SnowballC)
library(caret)
library(topicmodels)
library(tidytext)
library(cluster)
library(SentimentAnalysis)

syberia <- fromJSON("Syberia.json", flatten=TRUE)
syberia1.orig <- syberia %>% filter(product_id == "46500")
syberia2.orig <- syberia %>% filter(product_id == "46510")

syberia1.orig %>% nrow
syberia2.orig %>% nrow

# tdm and dtm
createTdmDtm <- function(text) {
  
  toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

  corp <- Corpus(VectorSource(text))
  corp <- tm_map(corp, toSpace, "/")
  corp <- tm_map(corp, toSpace, "@")
  corp <- tm_map(corp, toSpace, "\\|")
  corp <- tm_map(corp, toSpace, "[^\x01-\x7F]")
  corp <- tm_map(corp, toSpace, "¦")
  corp <- tm_map(corp, stripWhitespace)
  corp <- tm_map(corp, removePunctuation)
  corp <- tm_map(corp, removeNumbers)
  corp <- tm_map(corp, content_transformer(tolower))
  corp <- tm_map(corp, removeWords, stopwords("english"))
  corp <- tm_map(corp, stemDocument)
  
  dtm <- DocumentTermMatrix(corp)
  tdm <- TermDocumentMatrix(corp)
  
  list(dtm, tdm)
}

syberia1.DTM <- createTdmDtm(syberia1.orig$text)[[1]]
syberia1.TDM <- createTdmDtm(syberia1.orig$text)[[2]]

syberia2.DTM <- createTdmDtm(syberia2.orig$text)[[1]]
syberia2.TDM <- createTdmDtm(syberia2.orig$text)[[2]]

#Quick peek at the most common words
head(sort(rowSums(as.matrix(syberia1.TDM)), T), 10)
head(sort(rowSums(as.matrix(syberia2.TDM)), T), 10)

mostCommonWordsPlt <- function(DTM, sparsity = 0.9, gameName){
  
  freq <- colSums(as.matrix(removeSparseTerms(DTM, sparsity)))
  
  data.frame(word = names(freq), freq = freq) %>% 
    subset(., freq > 30) %>%
    ggplot(aes(x = reorder(word, -freq), freq)) + 
    geom_bar(stat = "identity", fill = "lightblue") + 
    theme(axis.text.x = element_text(angle=45, hjust=1)) + 
    labs(x = "Words", y = "Frequencies", title = paste0("Words frequencies for ", gameName))
  
}

syberia1.mostCommonPlot <- mostCommonWordsPlt(syberia1.DTM, 0.85, gameName = "Syberia 1")
syberia2.mostCommonPlot <- mostCommonWordsPlt(syberia2.DTM, 0.85, gameName = "Syberia 2")


findAssocs(syberia1.DTM, 
           c("graphic", "sound", "good", "bad", "stori", "audio", "plot", "quest"),
           corlimit = 0.4)
findAssocs(syberia2.DTM, 
           c("graphic", "sound", "good", "bad", "stori", "audio", "plot", "quest"),
           corlimit = 0.4) 

#### Models for classification #### 

syberia1.df <- cbind(as.data.frame(data.matrix(syberia1.DTM), stringsAsfactors = FALSE), as.factor(syberia1.orig$recommended))
syberia2.df <- cbind(as.data.frame(data.matrix(syberia2.DTM), stringsAsfactors = FALSE), as.factor(syberia2.orig$recommended))
colnames(syberia1.df)[ncol(syberia1.df)] <- "Recommended"
colnames(syberia2.df)[ncol(syberia2.df)] <- "Recommended"

set.seed(2137)
syberia1.part <- createDataPartition(syberia1.df$Recommended, p = 0.8, list = F)
syberia2.part <- createDataPartition(syberia2.df$Recommended, p = 0.8, list = F)

syberia1.train <- syberia1.df[syberia1.part, c(colSums(syberia1.df[, -ncol(syberia1.df)]) > 0.8*0.025*nrow(syberia1.df), T)]
syberia1.test <- syberia1.df[-syberia1.part, c(colSums(syberia1.df[, -ncol(syberia1.df)]) > 0.8*0.025*nrow(syberia1.df), T)]

syberia2.train <- syberia2.df[syberia2.part, c(colSums(syberia2.df[, -ncol(syberia2.df)]) > 0.8*0.025*nrow(syberia2.df), T)]
syberia2.test <- syberia2.df[-syberia2.part, c(colSums(syberia2.df[, -ncol(syberia2.df)]) > 0.8*0.025*nrow(syberia2.df), T)]


syberia1.train <- upSample(syberia1.train, syberia1.train$Recommended)
syberia2.train <- upSample(syberia2.train, syberia2.train$Recommended)

syberia1.train$Class <- NULL
syberia2.train$Class <- NULL

knn.grid <-  expand.grid(k = 10:20)

# SYBERIA 1
knn.model1 <- train(Recommended ~ . ,
                    data = syberia1.train,
                    method = "knn",
                    tuneGrid = knn.grid)

# SYBERIA 2
knn.model2 <- train(Recommended ~ . ,
                   data = syberia2.train,
                   method = "knn",
                   tuneGrid = knn.grid)


syberia1.knn.pred <- predict(knn.model1, syberia1.test)
syberia2.knn.pred <- predict(knn.model2, syberia2.test)

confusionMatrix(syberia1.test$Recommended, syberia1.knn.pred)
confusionMatrix(syberia2.test$Recommended, syberia2.knn.pred)

#----------------------------------

# SYBERIA 1
svm.model1 <- train(Recommended ~ . ,
                    data = syberia1.train,
                    method = "svmLinear",
                    preProcess = c("center", "scale"))

# SYBERIA 2
svm.model2 <- train(Recommended ~ . ,
                    data = syberia2.train,
                    method = "svmLinear",
                    preProcess = c("center", "scale"))


syberia1.svm.pred <- predict(svm.model1, syberia1.test)
syberia2.svm.pred <- predict(svm.model2, syberia2.test)

confusionMatrix(syberia1.test$Recommended, syberia1.svm.pred)
confusionMatrix(syberia2.test$Recommended, syberia2.svm.pred)

#----------------------------------

# SYBERIA 1
nb.model1 <- train(Recommended ~ . ,
                    data = syberia1.train,
                    method = "nb",
                    preProcess = c("center", "scale"))

# SYBERIA 2
nb.model2 <- train(Recommended ~ . ,
                    data = syberia2.train,
                    method = "nb",
                    preProcess = c("center", "scale"))


syberia1.nb.pred <- predict(nb.model1, syberia1.test)
syberia2.nb.pred <- predict(nb.model2, syberia2.test)

confusionMatrix(syberia1.test$Recommended, syberia1.nb.pred)
confusionMatrix(syberia2.test$Recommended, syberia2.nb.pred)

#### Clustering #### 

hClustTM <- function(TDM, k) {
  
  d <- dist(removeSparseTerms(TDM, 0.85), method = "manhattan")
  fit <- hclust(d, method = "ward.D2")    
  plot.new()
  plot(fit, hang=-1)
  groups <- cutree(fit, k)  
  rect.hclust(fit, k, border = "darkblue")
}

hClustTM(syberia1.TDM, 6)
hClustTM(syberia2.TDM, 6)

kMeansTM <- function(TDM, k) {
  
  d <- dist(removeSparseTerms(TDM, 0.85), method = "manhattan")
  
  kfit <- kmeans(d, k)
  clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines = 0)
  
}

kMeansTM(syberia1.TDM, 6)
kMeansTM(syberia2.TDM, 6)

#### Topic modelling #### 

top_terms_by_topic_LDA <- function(DTM, # columm from a dataframe
                                   plot = T, 
                                   number_of_topics = 4) # number of topics 
{    
  unique_indexes <- unique(DTM$i) # index of each unique value
  DTM <- DTM[unique_indexes,] 
  
  # preform LDA
  lda <- LDA(DTM, k = number_of_topics, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  # top ten terms for each topic
  top_terms <- topics  %>% 
    group_by(topic) %>% # each topic as a different group
    top_n(10, beta) %>% # 10 most informative words
    ungroup() %>% 
    arrange(topic, -beta) # arrange words 
  
  if(plot == T){
    top_terms %>% 
      mutate(term = reorder(term, beta)) %>% 
      ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
      geom_col(show.legend = FALSE) + # bar plot
      facet_wrap(~ topic, scales = "free") + 
      labs(x = NULL, y = "Beta") + 
      coord_flip() 
  } else{ 
    # list of sorted terms instead of a plot
    return(top_terms)
  }
}

top_terms_by_topic_LDA(syberia1.DTM, number_of_topics = 2, plot = T)
top_terms_by_topic_LDA(syberia2.DTM, number_of_topics = 2, plot = T)


GibbsLDA <- function(DTM, k=2) {

  burnin <- 4000
  iter <- 2000
  thin <- 500
  seed <-list(2003,5,63,100001,765)
  nstart <- 5
  unique_indexes <- unique(DTM$i)
  
  ldaOut <- LDA(DTM[unique_indexes, ], k, 
                method="Gibbs", 
                control=list(nstart=nstart, seed = seed, best=T, burnin = burnin, iter = iter, thin=thin))
  ldaOut.topics <- as.matrix(topics(ldaOut))
  ldaOut.terms <- as.matrix(terms(ldaOut,3))
  
  list(ldaOut.topics,
       ldaOut.terms)
  
}

syberia1.LDA <- GibbsLDA(syberia1.DTM)
syberia2.LDA <- GibbsLDA(syberia2.DTM)



#### Sentiment modelling #### 


sentiment <- analyzeSentiment(syberia1.orig$text)
plotSentiment(sentiment$SentimentHE)
convertToDirection(sentiment$SentimentQDAP)
data(DictionaryGI)

df <- data.frame(sentence=1:351,QDAP=sentiment$SentimentQDAP, GI=sentiment$SentimentGI)

ggplot(df, aes(x=sentence, y=QDAP)) + geom_line(color="red")+geom_line(aes(x=sentence, y=GI),color="green")




#na chart

tidy_books <- syberia1.orig %>%
  unnest_tokens(word, text)

get_sentiments("nrc")[, 2] %>% unique %>% pull %>%  lapply(function(x) {
  tidy_books %>%
    inner_join(get_sentiments("nrc") %>% 
                 filter(sentiment == x)) %>%
    count(word, sort = TRUE)
  
})

library(stringr)
library(OneR)


syberia1.orig$gameplay <- bin(syberia1.orig$hours, nbins = 5, na.omit = F)

syberia1.words <- syberia1.orig %>%
  unnest_tokens(word, text) %>%
  filter(str_detect(word, "[a-z']$"),
         !word %in% stop_words$word)

syberia1.words %>%
  count(word, sort = TRUE)

words_by_recommend <- syberia1.words %>%
  count(recommended, word, sort = TRUE) %>%
  ungroup()

words_by_gameplay <- syberia1.words %>%
  count(gameplay, word, sort = TRUE) %>%
  ungroup()

words_by_date <- syberia1.words %>%
  count(date, word, sort = TRUE) %>%
  ungroup()


words_by_recommend %>% filter(!recommended)

tf_idf <- words_by_recommend %>%
  bind_tf_idf(word, recommended, n) %>%
  arrange(desc(tf_idf))

tf_idf

#to jest najs

tf_idf %>%
  group_by(recommended) %>%
  top_n(12, tf_idf) %>%
  ungroup() %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(word, tf_idf, fill = recommended)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ recommended, scales = "free") +
  ylab("tf-idf") +
  coord_flip()



recommendp_sentiments <- words_by_recommend %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(recommended) %>%
  summarize(score = sum(score * n) / sum(n))

recommendp_sentiments %>%
  mutate(recommended = reorder(recommended, score)) %>%
  ggplot(aes(recommended, score, fill = score > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("Average sentiment score")

recommendp_sentiments <- words_by_gameplay %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(gameplay) %>%
  summarize(score = sum(score * n) / sum(n))

recommendp_sentiments %>%
  mutate(gameplay = reorder(gameplay, score)) %>%
  ggplot(aes(gameplay, score, fill = score > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("Average sentiment score")


contributions <- syberia1.words %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(word) %>%
  summarize(occurences = n(),
            contribution = sum(score))

contributions
contributions %>%
  top_n(25, abs(contribution)) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip()



## WORDCLOUD
# Do the word clouds
library(wordcloud)   
dtms <- removeSparseTerms(dtm, 0.15)   
freq <- colSums(as.matrix(dtm))   
wordcloud(names(freq),
          freq,
          scale = c(4,1),
          colors = brewer.pal(3, "Pastel2"),
          rot.per = 0)


#bigram

bigrams <- syberia1.orig %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)
trigrams <- syberia1.orig %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3)


bigrams %>%
  count(bigram, sort = TRUE)

trigrams %>%
  count(trigram, sort = TRUE)

bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word) %>%
  count(word1, word2, sort = TRUE)

trigrams %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  count(word1, word2, word3, sort = TRUE)


bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word) %>%
  count(word1, word2, sort = TRUE) %>%
  unite("bigram", c(word1, word2), sep = " ") %>%
  top_n(10) %>%
  ungroup() %>%
  ggplot(aes(bigram, n)) +
  geom_bar(stat = "identity", alpha = .8, show.legend = FALSE) +
  drlib::scale_x_reordered() +
  coord_flip()

bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(word1 == "not") %>%
  count(word1, word2, sort = TRUE)

trigrams %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(word1 == "not") %>%
  count(word1, word2, word3, sort = TRUE)


AFINN <- get_sentiments("afinn")

(nots <- bigrams %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(word1 == "not") %>%
    inner_join(AFINN, by = c(word2 = "word")) %>%
    count(word2, score, sort = TRUE) 
)

nots %>%
  mutate(contribution = n * score) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  ggplot(aes(reorder(word2, contribution), n * score, fill = n * score > 0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  xlab("Words preceded by 'not'") +
  ylab("Sentiment score * # of occurrances") +
  coord_flip()


negation_words <- c("not", "no", "never", "without")

(negated <- bigrams %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(word1 %in% negation_words) %>%
    inner_join(AFINN, by = c(word2 = "word")) %>%
    count(word1, word2, score, sort = TRUE) %>%
    ungroup()
)


#cors
#
(ps_words <- syberia1.orig %>% 
    unnest_tokens(word, text) %>%
    filter(!word %in% stop_words$word))

library(widyr)

#to na pozneij
(word_pairs <- ps_words %>%
    pairwise_count(word, recommended, sort = TRUE))
