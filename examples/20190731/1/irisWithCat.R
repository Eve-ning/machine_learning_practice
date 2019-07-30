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
