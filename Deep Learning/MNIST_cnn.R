# CNN
# Data Preparation -----------------------------------------------------

library(keras)
batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Model Architecture
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# Model config
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)
summary(model)

# Start Fitting
system.time({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2
  )
})

plot(history)

scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')

# Save and Load Model
# model %>% save_model_hdf5("MNIST_cnn.h5")
# model <- load_model_hdf5("MNIST_cnn.h5")
# new_model %>% summary()

# Save and Load Model weights
# model %>% save_model_weights_hdf5("MNIST_cnn_weights.h5")
# model %>% load_model_weights_hdf5("MNIST_cnn_weights.h5")
# model %>% predict_classes(x_test) %>% head()

probe <- array_reshape(x_test[sample(1:1000, 1),,,], c(1, img_rows, img_cols, 1))
image(t(probe[1, 28:1, ,1]*255), useRaster=FALSE, axes=FALSE, col=grey(seq(0, 1, length = 256)))

model %>% predict(probe)
model %>% predict(probe) %>% plot(x=0:9, y = ., type = "l", col = "red", xlab = "class", ylab = "prob")