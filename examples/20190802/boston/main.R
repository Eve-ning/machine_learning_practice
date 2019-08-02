library(neuralnet)
library(MASS)
library(dplyr)
library(magrittr)
library(reshape2)
boston.train.index <- sample(1:nrow(Boston),
                             size = nrow(Boston) * 0.7)

boston.scale <- Boston %>% 
  melt() %>% 
  group_by(variable) %>% 
  summarise(scales = max(value) - min(value),
            zeros = min(value))

boston.norm <- Boston %>% 
  tibble::rowid_to_column() %>% 
  melt(id.vars = 1) %>% 
  merge(boston.scale, by = 'variable') %>% 
  mutate(value = (value - zeros) / scales) %>% 
  dcast(rowid ~ variable, value.var = 'value', fun.aggregate = first) %>% 
  select(-rowid)

boston.train <- boston.norm[boston.train.index,]
boston.test <- boston.norm[-boston.train.index,]

boston.model <- neuralnet(medv ~ .,
                          data = boston.train,
                          hidden = c(5,5,5,5,5),
                          threshold = 0.002,
                          lifesign = 'full',
                          linear.output = T)

plot(boston.model)

boston.test.predict <-
  data.frame(value.predict = neuralnet::compute(boston.model,boston.test[,1:13])$net.result,
             value.actual = boston.test$medv,
             variable = 'medv') %>% 
  merge(boston.scale, by = 'variable') %>%
  mutate(value.predict = value.predict * scales + zeros,
         value.actual = value.actual * scales + zeros,
         value.error.sq = (value.predict - value.actual) ** 2)

boston.test.error.rmsq <- (boston.test.predict %>% 
  summarise(value.error.rmsq = sqrt(mean(value.error.sq))))[[1]]
  
