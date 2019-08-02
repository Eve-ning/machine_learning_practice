scores.raw <- read.csv(fileEncoding = 'utf-16', "Projects/repos/vsrgtools_dataset/src/datasets/osumania_performance/2019_07_01_performance_mania_top/score_sample.csv")

library(neuralnet)
library(dplyr)
library(magrittr)
library(ggplot2)
library(reshape2)
library(ggdark)

scores.index <- sample(1:100000, size = 5000)
scores <- scores.raw[scores.index,] %>%
  rename(count200 = countkatu,
         count300g = countgeki) %>% 
  select(countmiss, count50, count100, count200, count300, count300g, score) 

# Normalize here
scores.scales <- scores %>% 
  melt(id.vars = 1) %>% 
  group_by(variable) %>% 
  summarise(scale = max(value) - min(value),
            zero = min(value))

scores.norm <- scores %>% 
  tibble::rowid_to_column() %>% 
  melt(id.vars = 1) %>% 
  merge(scores.scales, by = 'variable') %>% 
  mutate(value = (value - zero) / scale) %>% 
  dcast(rowid ~ variable, value.var = 'value', fun.aggregate = first) %>% 
  select(-rowid)

# Extract 4000 for training, 1000 for test
scores.train.index <- sample(1:5000, 4000)

scores.train <- scores.norm[scores.train.index,]
scores.test <- scores.norm[-scores.train.index,]

# Create model
scores.model <- neuralnet(score ~ ., data = scores.norm, hidden = c(5,5), threshold = 0.02,
                          lifesign = 'full')

# Unnormalize
scores.test.predict <-
  data.frame(value.predict = neuralnet::compute(scores.model, scores.test[,1:5])$net.result,
             value.actual = scores.test$score,
             variable = 'score') %>% 
  merge(scores.scales, by = 'variable') %>% 
  mutate(value.predict = value.predict * scale + zero,
         value.actual = value.actual * scale + zero,
         value.error = value.predict - value.actual)

# Plot
ggplot(scores.test.predict) + 
  aes(value.actual, value.predict,
      color = value.error) +
  geom_point() +
  scale_color_gradient(low="cyan", high="green") +
  dark_theme_minimal()
      
