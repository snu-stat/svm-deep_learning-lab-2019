library(keras)

# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 30

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)
dim(y_train)
x_train[1, , ]

image(t(x_train[1, 28:1,]), useRaster=TRUE, axes=FALSE, col=grey(seq(0, 1, length = 256)))

y_train[1]

dim(x_test)
dim(y_test)

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255
dim(x_train)
dim(x_test)

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
dim(y_train)
head(y_train)

# Model Architecture
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_classes, activation = 'softmax')
summary(model)

# Model config
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Start Fitting
system.time({
  history <- model %>% fit(
    x_train, y_train, 
    epochs = epochs, batch_size = batch_size, 
    validation_split = 0.2
  )
})

plot(history)

model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test) %>% head()
