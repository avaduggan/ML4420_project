# clear enviorment
rm(list = ls())

library(dplyr)

set.seed(42) # set seed

dir <- ("/Users/mckennaquam/Desktop/DS 4420/ML4420_project/data/")
df <- read.csv(paste0(dir, "athlete_events.csv"))

sort_athletics <- function(e) {
  e <- tolower(e)
  len <- NA

  if (any(sapply(c("jump", "vault"), function(event) grepl(event, e)))) {
    return("Jump")
  } else if (any(sapply(c("discus", "javelin", "shot put", "throw"), function(event) grepl(event, e)))) {
    return("Field")
  } else if (any(sapply(c("kilometre", "cross-country", "marathon"), function(event) grepl(event, e)))) {
    return("Long")
  } else if (grepl("\\b\\d+[\\d,]*\\s+meter\\b", e)) {
    
    match <- regexpr("(\\d[\\d,]*)\\s+mile", e, perl = TRUE)
    if (match != -1) {
      matched_text <- regmatches(e, match)
      number_str <- sub("\\s+mile", "", matched_text)
      len <- as.integer(gsub(",", "", number_str))
    } else {
      cat("ERROR:", e, "\n")
    }
    
    if (len <= 1) {
      return("Medium")
    } else {
      return("Long")
    }
    
  } else if (grepl("\\b\\d+[\\d,]*\\s+meter\\b", e)) {
    
    match <- regexpr("(\\d[\\d,]*)\\s+meter", e, perl = TRUE)
    if (match != -1) {
      matched_text <- regmatches(e, match)
      number_str <- sub("\\s+meter", "", matched_text)
      len <- as.integer(gsub(",", "", number_str))
    } else {
      cat("ERROR:", e, "\n")
    }
    
    if (len <= 400) {
      return("Short")
    } else if (len > 400 & len <= 1609.34) {
      return("Medium")
    } else {
      return("Long")
    }
    
  } else {
    # disgarding Decathalon, Pentathelon, Heptathlon, All Around
    return("Not Classed")
  }
}

sort_athletics_vec <- Vectorize(sort_athletics)
df <- df %>% filter(Sport == 'Athletics') %>%
  select(Sex, Age, Height, Weight, Event) %>%
  na.omit() %>%
  mutate(Sex_M = ifelse(Sex > "M", 1, 0),
         Sex_F = ifelse(Sex > "F", 1, 0)) %>%
  select(!Sex) %>%
  mutate(Event_Type = sapply(Event, sort_athletics))

df$Event_Type = sort_athletics(df$Event)

sort_athletics("Athletics Men's 4 x 100 metres Relay")

e<- tolower("Athletics Men's 4 x 100 metres Relay")
grepl("meters", e)
  