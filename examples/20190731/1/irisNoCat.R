require(neuralnet)
# Create a dataset for neural network to train on
model <- neuralnet(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
                   data = iris,
                   hidden = c(4,4),
                   threshold = 0.03,
                   lifesign = 'full')

print(model)
plot(model)

model$net.result
model$result.matrix
