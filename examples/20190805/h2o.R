library(h2o)

c1 <- h2o.init(max_mem_size = '2G',
               nthreads = 2,
               ip = 'localhost',
               port = 54321)

iris_d1 <- h2o.deeplearning(1:4, 5,as.h2o(iris),hidden = c(5,5),
                            export_weights_and_biases = T)
plot(iris_d1)
iris_d1

h2o.weights(iris_d1,)
h2o.weights(iris_d1,2)
plot(as.data.frame(h2o.weights(iris_d1, 1))[,1])
h2o.confusionMatrix(iris_d1)

