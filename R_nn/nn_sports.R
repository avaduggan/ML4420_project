rm(list = ls())

library(dplyr)

set.seed(42)

dir <- ("/Users/mckennaquam/Desktop/DS 4420/ML4420_project/data/")
df <- read.csv(paste0(dir, "athlete_events.csv"))

sports_for_predictions <- c("Cycling", "Fencing", "Gymnastics", "Rowing", "Swimming")

df_sports <- df %>% select(Sex, Age, Height, Weight, Sport) %>%
  filter(Sport %in% sports_for_predictions) %>%
  na.omit() %>%
  mutate(Sex_M = ifelse(Sex > "M", 1, 0),
         Sex_F = ifelse(Sex > "F", 1, 0)) %>%
  select(!Sex)

scale_no_center <- function(x, min, max) {
  (x - min) / (max - min)
}

df_sports$Age <- scale_no_center(df_sports$Age, min(df_sports$Age), max(df_sports$Age))
df_sports$Height <- scale_no_center(df_sports$Height, min(df_sports$Height), max(df_sports$Height))
df_sports$Weight <- scale_no_center(df_sports$Weight, min(df_sports$Weight), max(df_sports$Weight))


for (s in sports_for_predictions) {
  df_sports <- df_sports %>% mutate(!!s := ifelse(Sport == s, 1, 0))
}

df_sports <- df_sports %>% select(!Sport)

X <- df_sports %>% select(Age, Height, Weight, Sex_M, Sex_F) 
y <- df_sports %>% select(sports_for_predictions)


sample <- sample(c(TRUE, FALSE), nrow(df_sports), replace=TRUE, prob=c(0.7,0.3))

X_train <- data.matrix(X[sample,])
X_test <- data.matrix(X[!sample,])  
y_train <- data.matrix(y[sample, ])
y_test <- data.matrix(y[!sample,])

# -------------------
# begin neural network code

node_depth <- 4

# define weights
W1 <- matrix(rnorm(5 * node_depth), nrow = 5, ncol = node_depth)
W2 <- matrix(rnorm(node_depth * 5), nrow = node_depth, ncol = 5)

# Define the ReLU activation function
relu <- function(x) {
  matrix(pmax(0, x))
}

softmax <- function(x) {
  demonimator <- sum(exp(x))
  for (i in 1:length(x)) {
    x[i] <- exp(x[i]) / demonimator
  }
  return (x)
}

f <- function(x) {
  h <- relu(t(W1)%*%x)
  return(softmax(t(W2)%*%h))
}

errors <- numeric()
epochs <- 10
n <- nrow(X_train)
eta <- 0.01

softmax_derrivative <- function(h, y, W2) {
  # h is the hidden vector
  # y is the target
  # W2 is the output weights
  softmax_denominator <- sum(exp(t(W2) %*% h))
  
  dW2_h <- matrix(0, nrow = node_depth, ncol = 5)
  for (i in 1:node_depth) {
    for (j in 1:5) {
      dW2_h[i, j] <- -1*y[j]*h[i] - length(y)*((h[i]*exp(t(W2[,j])%*%h))/softmax_denominator) 
    }
  }
  return (dW2_h)
}

derivation_loss_h1 <- function(h, y, W2) {
  softmax_denominator <- sum(exp(t(W2) %*% h))
  dL_dh <- matrix(0, nrow=node_depth, ncol=1)
  for (i in 1:node_depth) {
    softmax_numerator <- sum(W2[i,] * exp(t(W2) %*% h))
    dL_dh[i] <- -1*sum(W2[i,]*y[i]) - length(y)*(softmax_numerator/softmax_denominator)
  }
  return(dL_dh)
}

relu_derrivative <- function(x, h){
  mat1 <- ifelse(h > 0, 1, 0) 
  dW_dh<- kronecker(t(mat1), x)
  return(dW_dh)
}

for (epoch in 1:epochs) {
  
  
  dW2 <- matrix(0, nrow = node_depth, ncol = 5)
  for (i in 1:n) {
    x <- matrix(X_train[i, ], nrow = 5, ncol = 1)
    h <- relu(t(W1) %*% x)
    dW2 <- dW2 + ((1/n) * softmax_derrivative(h, matrix(y_train[i,], nrow = 5, ncol=1), W2))
  }
  W2 <- W2 - eta * dW2
  
  
  dW1 <- matrix(0, nrow = 5, ncol = node_depth)
  for (i in 1:n) {
    x <- matrix(X_train[i, ], nrow = 5, ncol = 1)
    h <- relu(t(W1) %*% x)
    dLdh <- derivation_loss_h1(h, matrix(y_train[i,], nrow = 5), W2)
    dhdW <- relu_derrivative(x, h)
    mat1 <- matrix(1, nrow=5, ncol=1)
    dW1 <- dW1 + (1/n) * (mat1 %*% t(dLdh)) * dhdW
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

