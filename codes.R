library(jsonlite)
library(dplyr)
library(tm)
library(SnowballC)
library(caret)
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
syberia.train2 <- syberia.train[, c(colSums(syberia.train[, -ncol(syberia.train)]) > 0.1*nrow(syberia.train), T)]


knn.grid <-  expand.grid(k = 20:40)
knn.model <- train(Recommended ~ . ,
                   data = syberia.train2,
                   method = "knn",
                   tuneGrid = knn.grid)
