dataset <- ISLR::Auto
dataset.mean <- apply(dataset[,1:6], MARGIN = 2, FUN = mean)
dataset.sd <- apply(dataset[,1:6], MARGIN = 2, FUN = sd)

dataset.scaled <-
  as.data.frame(scale(dataset[,1:6],
                      center = dataset.mean,
                      scale = dataset.sd))
set.seed(1)
dataset.train.index <- sample(1:nrow(dataset.scaled),
                              size = 0.7 * nrow(dataset.scaled))
dataset.train <- dataset.scaled[dataset.train.index,]
dataset.test <- dataset.scaled[-dataset.train.index,]

# Neural Network
dataset.nn <- neuralnet::neuralnet(mpg ~ .,
                                   data = dataset.train,
                                   hidden = c(3),
                                   linear.output = T,
                                   lifesign = 'full')

dataset.test.predict <- neuralnet::compute(dataset.nn, dataset.test[,2:6])
dataset.test.mserror <-
  mean(sum((dataset.test.predict$net.result - dataset.test$mpg) ** 2))

# Linear Model
dataset.lm <- lm(mpg ~ ., data = dataset.train)
summary(dataset.lm)
dataset.lm.test.predict <- predict(dataset.lm, dataset.test[,1:6])
dataset.lm.test.mserror <-
  mean(sum((dataset.lm.test.predict - dataset.test$mpg) ** 2))

require(ggplot2)
ggplot() +
  geom_point(aes(x = dataset.test$mpg,
                 y = dataset.test.predict$net.result),
             color = 'red') +
  geom_point(aes(x = dataset.test$mpg,
                 y = dataset.lm.test.predict,
                 color = 'Linear Model'),
             color = 'blue') +
  geom_abline(intercept = 0)
  
