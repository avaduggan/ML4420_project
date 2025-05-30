# clear environment
rm(list = ls())

library(dplyr)

# set seed for reproducable results
set.seed(42) # set seed

dir <- ("/Users/mckennaquam/Desktop/DS 4420/ML4420_project/data/")
df_athletics <- read.csv(paste0(dir, "athletics_cleaned.csv"))

# data cleaned in python! (I couldnt get my sorting function to work in R)
# hot 1 encode event categories
events_for_prediction <- c("Short", "Mid", "Long", "Field", "Jump")
# hot 1 encode our sports into separate columns and remove sport columns
for (e in events_for_prediction) {
  df_athletics <- df_athletics %>% mutate(!!e := ifelse(event_catagory == e, 1, 0))
}
df_athletics <- df_athletics %>% select(!event_catagory)


# separate features from output
X <- df_athletics %>% select(Age, Height, Weight, Sex_M, Sex_F) 
y <- df_athletics %>% select(events_for_prediction)

# test train split
sample <- sample(c(TRUE, FALSE), nrow(df_athletics), replace=TRUE, prob=c(0.7,0.3))

X_train <- data.matrix(X[sample,])
X_test <- data.matrix(X[!sample,])  
y_train <- data.matrix(y[sample, ])
y_test <- data.matrix(y[!sample,])

node_depth <- 100 # depth of the hidden layer

# define weights
W1 <- matrix(rnorm(5 * node_depth), nrow = 5, ncol = node_depth)
W2 <- matrix(rnorm(node_depth * 5), nrow = node_depth, ncol = 5)

# definition of relu  function
relu <- function(x) {
  matrix(pmax(0, x))
}

# definition of softmax function
softmax <- function(x) {
  demonimator <- sum(exp(x))
  for (i in 1:length(x)) {
    x[i] <- exp(x[i]) / demonimator
  }
  return (x)
}

# forward pass through the network
f <- function(x) {
  h <- relu(t(W1)%*%x)
  return(softmax(t(W2)%*%h))
}

errors <- numeric()
epochs <- 100
n <- nrow(X_train) # number of samples
eta <- 0.05 # learning rate

for (epoch in 1:epochs) {
  
  # first step of backprop
  dW2 <- matrix(0, nrow = node_depth, ncol = 5)
  for (i in 1:n) {
    x <- matrix(X_train[i, ], nrow = 5, ncol = 1)
    h <- relu(t(W1) %*% x)
    dLdW2 <- h %*% t(f(x) - matrix(y_train[i,], nrow = 5, ncol=1))
    dW2 <- dW2 + (1/n) * dLdW2  
  }
  W2 <- W2 - eta * dW2
  
  # second step of backprop
  dW1 <- matrix(0, nrow = 5, ncol = node_depth)
  for (i in 1:n) {
    x <- matrix(X_train[i, ], nrow = 5, ncol = 1)
    h <- relu(t(W1) %*% x)
    mat1 <- ifelse(h > 0, 1, 0)
    dLdW1 <- x %*% t((W2 %*% (f(x) - matrix(y_train[i,], nrow=5, ncol=1))) * mat1)
    dW1 <- dW1 + (1/n) * dLdW1
  }
  W1 <- W1 - eta * dW1
  
  
  # Compute error
  e <- 0
  for (i in 1:n) {
    o <- f(matrix(X_train[i, ], nrow = 5, ncol = 1))
    e <- e + (1 / n) * -1 * sum(matrix(y_train[i,], nrow = 5, ncol = 1) * log(o))
  }
  
  errors <- c(errors, e)
  print(paste0("epoch ", epoch, ": ", e))
}

# saving the errors of training process
epoch_range <- c(1:epochs)
df_sports_training_errors <- data.frame(epoch_range, errors)
save_dir <- "/Users/mckennaquam/Desktop/DS 4420/ML4420_project/R_nn/"
write.csv(df_sports_training_errors, paste0(save_dir, "nn_outputs/athletics_errors.csv"))

# saving the weights to csvs in case I need them in another file
df_W1 <- data.matrix(W1)
write.csv(W1, paste0(save_dir, "nn_outputs/athletics_W1.csv"))
df_W2 <- data.matrix(W2)
write.csv(W2, paste0(save_dir, "nn_outputs/athletics_W2.csv"))


# running the nn for the test data
df_y_predicted <- data.frame(Short=numeric(0),
                             Mid=numeric(0),
                             Long=numeric(0),
                             Field=numeric(0),
                             Jump=numeric(0))
for (i in 1:nrow(X_test)) {
  y_predicted <- f(matrix(X_test[i,], nrow=5, ncol=1))
  df_y_predicted[nrow(df_y_predicted) + 1,] <- t(y_predicted)
}

df_y_test <- data.frame(y_test)

# saving the predictions and the true labels to a dataframe
predicted_labels <- colnames(df_y_predicted)[apply(df_y_predicted,1,which.max)]
test_labels <- colnames(df_y_test)[apply(df_y_test,1,which.max)]
df_sports_predictions <- data.frame(predicted_labels, test_labels)
write.csv(df_sports_predictions, paste0(save_dir, "nn_outputs/athletics_nn_results.csv"))

scratch <- df_sports_predictions %>% mutate(correct = (predicted_labels == test_labels)) %>%
  group_by(test_labels, correct) %>% count()
