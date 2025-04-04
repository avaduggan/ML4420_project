# clear envoirment
rm(list = ls())

library(dplyr)

set.seed(42) # set seed

# directory to load data from
dir <- ("/Users/mckennaquam/Desktop/DS 4420/ML4420_project/data/")
df <- read.csv(paste0(dir, "athlete_events.csv"))

# the sports we are going to try and predict (some of the most represented in our data)
sports_for_predictions <- c("Cycling", "Fencing", "Gymnastics", "Rowing", "Swimming")

# select columns which have our features (sex, age, height weight) and output (sport)
# filter the data frame for the sports we have selected
# remove any rows with missing data
# hot 1 encode sex as Sex_M and Sex_F (drop sex column)
df_sports <- df %>% select(Sex, Age, Height, Weight, Sport) %>%
  filter(Sport %in% sports_for_predictions) %>%
  na.omit() %>%
  mutate(Sex_M = ifelse(Sex > "M", 1, 0),
         Sex_F = ifelse(Sex > "F", 1, 0)) %>%
  select(!Sex)

# scale the numeric columns from 0-1
scale_no_center <- function(x, min, max) {
  (x - min) / (max - min)
}
# numeric features of age, height, and weight
df_sports$Age <- scale_no_center(df_sports$Age, min(df_sports$Age), max(df_sports$Age))
df_sports$Height <- scale_no_center(df_sports$Height, min(df_sports$Height), max(df_sports$Height))
df_sports$Weight <- scale_no_center(df_sports$Weight, min(df_sports$Weight), max(df_sports$Weight))

# hot 1 encode our sports into separate columns and remove sport columns
for (s in sports_for_predictions) {
  df_sports <- df_sports %>% mutate(!!s := ifelse(Sport == s, 1, 0))
}
df_sports <- df_sports %>% select(!Sport)

# seperate features from output
X <- df_sports %>% select(Age, Height, Weight, Sex_M, Sex_F) 
y <- df_sports %>% select(sports_for_predictions)

# test train split
sample <- sample(c(TRUE, FALSE), nrow(df_sports), replace=TRUE, prob=c(0.7,0.3))

X_train <- data.matrix(X[sample,])
X_test <- data.matrix(X[!sample,])  
y_train <- data.matrix(y[sample, ])
y_test <- data.matrix(y[!sample,])

# -------------------
# begin neural network code

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
write.csv(df_sports_training_errors, paste0(save_dir, "nn_outputs/sports_errors.csv"))

# saving the weights to csvs in case I need them in another file
df_W1 <- data.matrix(W1)
write.csv(W1, paste0(save_dir, "nn_outputs/sports_W1.csv"))
df_W2 <- data.matrix(W2)
write.csv(W2, paste0(save_dir, "nn_outputs/sports_W2.csv"))

# running the nn for the test data
df_y_predicted <- data.frame(Cycling=numeric(0),
                             Fencing=numeric(0),
                             Gymnastics=numeric(0),
                             Rowing=numeric(0),
                             Swimming=numeric(0))
for (i in 1:nrow(X_test)) {
  y_predicted <- f(matrix(X_test[i,], nrow=5, ncol=1))
  df_y_predicted[nrow(df_y_predicted) + 1,] <- t(y_predicted)
}

df_y_test <- data.frame(y_test)

# saving the predictions and the true labels to a dataframe
predicted_labels <- colnames(df_y_predicted)[apply(df_y_predicted,1,which.max)]
test_labels <- colnames(df_y_test)[apply(df_y_test,1,which.max)]
df_sports_predictions <- data.frame(predicted_labels, test_labels)
write.csv(df_sports_predictions, paste0(save_dir, "nn_outputs/sports_nn_results.csv"))