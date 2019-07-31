# We need to process iris' categorical dataset
require(dplyr)
require(magrittr)
require(neuralnet)
require(nnet)
require(NeuralNetTools)
require(tidyr)

iris.new <- cbind(iris[, 1:4], class.ind(iris$Species))

iris.nn <- nnet(setosa + versicolor + virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
     iris.new,rang = 0.1, size = 5, decay = 0.1, maxit = 5000)
print(iris.nn)
plotnet(iris.nn)
garson(iris.nn)

iris.nn2 <- neuralnet(setosa + versicolor + virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                      iris.new, hidden = c(5,5),threshold = 0.01, lifesign = 'full')

plot(iris.nn2)
iris.nn2$covariate # Importance of input
iris.nn2$weights # Weights, Bias first then the following neurons
iris.nn2$net.result # Classification

# Create a confusion matrix to tabulate results

require(caret)
require(reshape2)

net.result <-
  as.data.frame(iris.nn2$net.result) %>% 
  set_colnames(c("setosa", "versicolor", "virginica")) %>% 
  tibble::rowid_to_column() %>% 
  melt(id.vars = 1, variable.name = "species", value.name = "probability") %>% 
  group_by(rowid) %>% 
  mutate(selected = max(probability) == probability) %>% 
  filter(selected == TRUE)

# 100% Accuracy
confusionMatrix(net.result$species, iris$Species)
