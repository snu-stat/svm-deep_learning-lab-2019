library(keras)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)

# Parameters --------------------------------------------------------------

maxlen <- 40

# Data Preparation --------------------------------------------------------

# Retrieve text
#path <- get_file(
#  'nietzsche.txt', 
#  origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
#)

path <- "nietzsche.txt"
read_lines(path) %>% head()

text <- read_lines(path) %>%
  str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)

print(sprintf("corpus length: %d", length(text)))

chars <- text %>%
  unique() %>%
  sort()

chars

dataset <- map(
  seq(1, length(text) - maxlen - 1, by = 3), 
  ~list(sentece = text[.x:(.x + maxlen - 1)], next_char = text[.x + maxlen])
)

dataset <- transpose(dataset)
dataset$sentece[[1]]
dataset$sentece[[2]]
dataset$next_char[[1]]
dataset$next_char[[2]]

print(sprintf("total chars: %d", length(chars)))

X <- array(0, dim = c(length(dataset$sentece), maxlen, length(chars)))
y <- array(0, dim = c(length(dataset$sentece), length(chars)))

for(i in 1:length(dataset$sentece)){
  
  X[i, , ] <- sapply(chars, function(x){
    as.integer(x == dataset$sentece[[i]])
  })
  
  y[i, ] <- as.integer(chars == dataset$next_char[[i]])
  
}

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 128, input_shape = c(maxlen, length(chars))) %>%
  layer_dense(length(chars)) %>%
  layer_activation("softmax")

optimizer <- optimizer_rmsprop(lr = 0.01)

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)
summary(model)

sample_mod <- function(preds, temperature = 1) {
  preds <- log(preds) / temperature
  exp_preds <- exp(preds)
  preds <- exp_preds / sum(exp(preds))
  
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}

diversity <- 0.5
set.seed(123)

system.time({
  for(iteration in 1:20){
    cat(sprintf("iteration: %02d ---------------\n\n", iteration))
    model %>% fit(
      X, y,
      batch_size = 128,
      epochs = 1
    )
    
    start_index <- sample(1:(length(text) - maxlen), size = 1)
    sentence <- text[start_index:(start_index + maxlen - 1)]
    generated <- ""
    
    for(i in 1:400){
      
      x <- sapply(chars, function(x){
        as.integer(x == sentence)
      })
      x <- array_reshape(x, c(1, dim(x)))
      
      preds <- predict(model, x)
      next_index <- sample_mod(preds, diversity)
      next_char <- chars[next_index]
      
      generated <- str_c(generated, next_char, collapse = "")
      sentence <- c(sentence[-1], next_char)
      
    }
    
    cat(generated)
    cat("\n\n")
  }
})
