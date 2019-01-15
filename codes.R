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


syberia1 <- Corpus(VectorSource(syberia1.orig$text))
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
syberia1 <- tm_map(syberia1, toSpace, "/")
syberia1 <- tm_map(syberia1, toSpace, "@")
syberia1 <- tm_map(syberia1, toSpace, "\\|")
syberia1 <- tm_map(syberia1, toSpace, "???")
syberia1 <- tm_map(syberia1, toSpace, "???")
syberia1 <- tm_map(syberia1, toSpace, "¦")

syberia1 <- tm_map(syberia1, stripWhitespace)

# remove unnecessary punctuation
syberia1 <- tm_map(syberia1, removePunctuation)

# remove unnecessary numbers
syberia1 <- tm_map(syberia1, removeNumbers)
syberia1 <- tm_map(syberia1, content_transformer(tolower))

syberia1 <- tm_map(syberia1, removeWords, stopwords("english"))
syberia1 <- tm_map(syberia1, stemDocument)


dtm <- DocumentTermMatrix(syberia1)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)


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

# Do the word clouds
library(wordcloud)   
dtms <- removeSparseTerms(dtm, 0.15)   
freq <- colSums(as.matrix(dtm))   
dark2 <- brewer.pal(6, "Dark2")   
wordcloud(names(freq), freq, max.words=100, rot.per=0.2, colors=dark2) 



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

top_terms_by_topic_LDA(syberia2$text, number_of_topics = 4, plot = T)



