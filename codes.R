library(jsonlite)
library(dplyr)
library(tm)
library(SnowballC)
library(caret)
library(topicmodels)
library(tidytext)

# sentiment == recommended?
# sentiment over time?
# model weighted by hours/products
# recommended as a label

syberia <- fromJSON("Syberia.json", flatten=TRUE)
syberia1.orig <- syberia %>% filter(product_id == "46500")
syberia2.orig <- syberia %>% filter(product_id == "46510")

syberia1.orig %>%  nrow
syberia2.orig %>%  nrow

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

####
#### Model
####

mat.df <- as.data.frame(data.matrix(dtm), stringsAsfactors = FALSE)
mat.df <- cbind(mat.df, as.factor(syberia1.orig$recommended))
colnames(mat.df)[ncol(mat.df)] <- "Recommended"

syberia.part <- createDataPartition(mat.df$Recommended, p = 0.8, list = F)
syberia.train <- mat.df[syberia.part, ]
syberia.test <- mat.df[-syberia.part, ]

syberia.train <- syberia.train[ ,-c(which(colnames(syberia.train) == "valadilen"))]
syberia.train2 <- syberia.train[, c(colSums(syberia.train[, -ncol(syberia.train)]) > 0.025*nrow(syberia.train), T)]
syberia.train3 <- upSample(syberia.train2, syberia.train2$Recommended)
syberia.test$Class <- syberia.test$Recommended


knn.grid <-  expand.grid(k = 10:20)
knn.model <- train(Recommended ~ . ,
                   data = syberia.train3,
                   method = "knn",
                   tuneGrid = knn.grid)

predictions <- predict(knn.model, syberia.test)

## 92% acc najssss
confusionMatrix(syberia.test$Class, predictions)



svm.model <- train(Recommended ~ . ,
                   data = syberia.train3,
                   method = "svmLinear",
                   preProcess = c("center", "scale"))
## 100% XDDDDDDDDDDDDDDDD
predictions <- predict(svm.model, syberia.test)
confusionMatrix(syberia.test$Class, predictions)

## 100% XDDDDDDDDDDDDDDDD
nb.model <-  train(Recommended ~ . ,
             data = syberia.train3,
             method = "nb",
             preProcess = c("center", "scale"))

predictions.nb <- predict(nb.model, syberia.test)
confusionMatrix(syberia.test$Class, predictions)

## CLUSTERING

freq <- colSums(as.matrix(removeSparseTerms(dtm, 0.90))) 
freq   

wf <- data.frame(word=names(freq), freq=freq)   
p <- ggplot(subset(wf, freq>20), aes(reorder(word, -freq), freq))    
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
p 

findAssocs(dtm, c("graphic" , "sound", "good", "stori"), corlimit=0.3) 





##HCLUST

dtms <- removeSparseTerms(dtm, 0.9)
library(cluster)   
d <- dist(t(dtms), method="euclidian")   
fit <- hclust(d=d, method="complete")    
plot.new()
plot(fit, hang=-1)
groups <- cutree(fit, k=6)  
rect.hclust(fit, k=6, border="red") 


kfit <- kmeans(d, 4)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)
wss <- 2:29
for (i in 2:29) wss[i] <- sum(kmeans(d,centers=i,nstart=25)$withinss)
plot(2:29, wss[2:29], type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")



## TOPIC
## 
## 


top_terms_by_topic_LDA <- function(input_text, # columm from a dataframe
                                   plot = T, 
                                   number_of_topics = 4) # number of topics 
{    
  # corpus and document term matrix
  Corpus <- Corpus(VectorSource(input_text)) 
  DTM <- DocumentTermMatrix(Corpus) 
  
  # remove empty rows in the dtm, necessary to perform LDA 
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
  }else{ 
    # list of sorted terms instead of a plot
    return(top_terms)
  }
}

top_terms_by_topic_LDA(syberia1.orig$text, number_of_topics = 4, plot = T)


# sENTIMENT
# 
# 

library(SentimentAnalysis)
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
