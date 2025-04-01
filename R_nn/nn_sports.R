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

X <- df_sports %>% select(Age, Height, Weight, Sex_M, Sex_F) %>% pull()
y <- df_sports %>% select(sports_for_predictions)

sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.7,0.3))
train  <- df[sample, ]
test   <- df[!sample, ]

#df_sports %>% count(Sport)
