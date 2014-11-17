#Load the libraries
library(tm)
library(lubridate)
library(ggplot2)
library(reshape2)
library(class)


options(stringsAsFactors=FALSE)
textData <- read.csv("~/DroneProject/DroneArticleData.txt", sep="^", header=FALSE, quote = "")
sentimentData <- read.csv("~/DroneProject/AFINN-111.txt", sep="\t", header=FALSE)
textData[,6] <- NULL
colnames(textData) <- c("newspaper", "date", "wordcount", "title", "body")
textData[,5] <- sapply(textData[,5], function(x){removePunctuation(x, preserve_intra_word_dashes = TRUE)})
corp <- VCorpus(VectorSource(textData[,5]))
corp2 <- tm_map(corp, stripWhitespace)
corp3 <- tm_map(corp2, content_transformer(tolower))
corp4 <- tm_map(corp3, removeWords, stopwords("english"))
save(file = "~/DroneProject/FinalCorpus.Rds", corp4)
load(file = "~/DroneProject/FinalCorpus.Rds")

plainText <- as.list(corp4)

dtm <- DocumentTermMatrix(corp4)
dtm2 <- removeSparseTerms(dtm, .75)

wordMatrix <- as.data.frame(as.matrix(dtm2))
wordMatrix2 <- as.data.frame(lapply(wordMatrix, function(x){x/rowSums(wordMatrix)}))

trainingSet <- read.csv("~/DroneProject/TrainingSet2.csv")

trainingSet2 <- melt(trainingSet)
trainingSet3 <- trainingSet2[-which(is.na(trainingSet2[,2])),]


removeRows <- c()
for(i in trainingSet3[,2]){
    print(i)
    vec1 <- as.numeric(as.vector(wordMatrix2[i,]))
    for(j in 1:nrow(wordMatrix2)){
        if(i!=j & !(j %in% trainingSet3[,2])){
            vec2 <- as.numeric(as.vector(wordMatrix2[j,]))
            if(cor(vec1, vec2) > .95 | textData[i,4] == textData[j,4]){
                if(!(j %in% trainingSet3[,2])){
                    print(paste("Row", j, "removed"))
                    removeRows <- append(removeRows, j)
                }
            }
        }
    }
}

wordMatrix3 <- wordMatrix2[-c(removeRows, trainingSet3[,2]),]

removeRows1 <- c()
for(i in 1:nrow(wordMatrix3)){
    print(i)
    if(!(i %in% removeRows1)){
        vec1 <- as.numeric(as.vector(wordMatrix3[i,]))
        for(j in (i+1):nrow(wordMatrix3)){
            
            vec2 <- as.numeric(as.vector(wordMatrix3[j,]))
            if(cor(vec1, vec2) > .95 | textData[i,4] == textData[j,4]){
                if(!(row.names(wordMatrix3)[j] %in% trainingSet3[,2])){
                    print(paste("Row", j, "removed"))
                    removeRows1 <- append(removeRows1, row.names(wordMatrix3)[j])
                }
            }
        }
    }
}

removeRows <- c(removeRows, removeRows1)
removeRows <- as.numeric(removeRows)
save(file = "~/DroneProject/RemoveRows.Rds", removeRows)
load(file = "~/DroneProject/RemoveRows.Rds")
removeRows <- removeRows[!is.na(removeRows)]
# load(file = "~/DroneProject/RemoveRows.Rds")
textData2 <- textData[-removeRows,]
plainText2 <- plainText[-removeRows]
wordMatrix4 <- wordMatrix2[-removeRows,]

sentiment <- c()
occurances <- c()
for(i in 1:length(plainText2)){
    print(i)
    string <- as.character(plainText2[[i]])
    string <- unlist(strsplit(string, " "))
    scores <- c()
    count <- 0
    for(j in string){
        if(j == "drone" | j == "drones"){
            count <- count+1
        }
        if(j %in% as.character(sentimentData[,1])){
            scores <- append(scores, sentimentData[which(sentimentData[,1]==j), 2])
        }
    }
    sentiment <- append(sentiment, sum(scores))
    occurances <- append(occurances, count)
}

finalData <- cbind(textData2, sentiment, occurances)
finalData$date <- as.Date(finalData$date, format="%B %d, %Y")

for(i in 1:nrow(finalData)){
    article <- as.numeric(row.names(finalData[i,]))
    finalData$category[i] <- ifelse(article %in% trainingSet3[,2], as.character(trainingSet3[which(trainingSet3[,2] == article),1]), NA)
}

length(intersect(as.numeric(row.names(finalData)), trainingSet3[,2]))

wordMatrixCat <- wordMatrix4[which(!is.na(finalData$category)),]

set.seed(0)
train <- sample(1:99, 50)
test <- setdiff(1:99, train)
wordMatrixTrain <- wordMatrixCat[train,]
wordMatrixTest <- wordMatrixCat[test,]

colnames(wordMatrixTrain) <- make.names(colnames(wordMatrixTrain))

trainingCats <- ifelse(finalData[which(!is.na(finalData$category)),]$category[train]=="Junk", "Junk", "Not Junk")
testingCats <- ifelse(finalData[which(!is.na(finalData$category)),]$category[test]=="Junk", "Junk", "Not Junk")

trainingCats <- finalData[which(!is.na(finalData$category)),]$category[train]
testingCats <- finalData[which(!is.na(finalData$category)),]$category[train]

for(i in 1:15){
    results <- as.character(knn(wordMatrixTrain, wordMatrixTest, trainingCats, k = i))
    accurate <- testingCats == results
    print(paste(i, ":", round((sum(accurate)/length(accurate))*100, 2)))
}

for(i in 1:15){
    results <- as.character(knn(wordMatrixTrain, wordMatrixTrain, trainingCats, k = i))
    accurate <- trainingCats == results
    print(paste(i, ":", round((sum(accurate)/length(accurate))*100, 2)))
}

results <- as.character(knn(wordMatrixTrain, wordMatrix4, trainingCats, k = 15))

wordMatrix5 <- wordMatrix4[results!="Junk",]

wordMatrixCat2 <- wordMatrix4[which((!is.na(finalData$category)) & finalData$category!="Junk"),]

set.seed(0)
train <- sample(1:59, 30)
test <- setdiff(1:59, train)
wordMatrixTrain2 <- wordMatrixCat[train,]
wordMatrixTest2 <- wordMatrixCat[test,]

trainingCats <- finalData[which(!is.na(finalData$category)),]$category[train]
testingCats <- finalData[which(!is.na(finalData$category)),]$category[test]

for(i in 1:15){
    results <- as.character(knn(wordMatrixTrain2, wordMatrixTest2, trainingCats, k = i))
    accurate <- testingCats == results
    print(paste(i, ":", round((sum(accurate)/length(accurate))*100, 2)))
}

results <- as.character(knn(wordMatrixTrain2, wordMatrixTest2, trainingCats, k = i))

# Sentiment over time graph

ggplot(finalData, aes(x=date, y=sentiment, colour=newspaper)) + geom_point(alpha=.5) + facet_grid(newspaper~.) + ylim(c(-100,100)) + stat_smooth() + xlab("Date") + ylab("Sentiment")

# Sentiment vs mentions

ggplot(finalData, aes(x=occurances, y=sentiment, colour=newspaper)) + geom_point(alpha=.5) + facet_grid(newspaper~.) + xlab('Mentions of "drone" or "drones"') + ylab("Sentiment")

summary(finalData$occurances)

plot(finalData$sentiment~finalData$occurances)
lm(finalData$sentiment~finalData$occurances)


lm(finalData$sentiment ~finalData$newspaper)

finalData$date


d <- dist(wordMatrix2, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward.D")
plot(fit) # display dendogram


hist(kmeans(mydata, centers=4)$cluster)


mydata <- wordMatrix2
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,
   centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
  ylab="Within groups sum of squares") 



findAssocs(dtm, "journalism", .6)

