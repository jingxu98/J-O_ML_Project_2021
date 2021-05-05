#calling packages
library(tidyverse)
library(dplyr)

#importing stroke dataset
setwd("/Users/oliviadelau/Desktop")
stroke_orig <- read.csv("healthcare-dataset-stroke-data.csv")
stroke_orig

summary(stroke_orig)


#removing rows with N/A values
stroke<- stroke_orig %>% dplyr::na_if("N/A")


stroke$smoking_status[stroke$smoking_status == "Unknown"] <- NA

stroke <- stroke %>% 
  na.omit() %>%
  filter(gender != "Other") %>%
  subset(select = -c(id))
stroke

#standardizing all string values to lowercase
stroke$gender <- tolower(stroke$gender)
stroke$work_type <- tolower(stroke$work_type)
stroke$Residence_type <- tolower(stroke$Residence_type)
stroke$smoking_status <- tolower(stroke$smoking_status)

#creating standard underscore between strings
stroke$work_type<- as.character(sub("-", "_", stroke$work_type))
stroke$smoking_status<- as.character(sub(" ", "_", stroke$smoking_status))


#changing all categorical variables to factors
stroke$gender <- factor(stroke$gender)
stroke$hypertension <- factor(stroke$hypertension)
stroke$heart_disease <- factor(stroke$heart_disease)
stroke$work_type <- factor(stroke$work_type)
stroke$Residence_type <- factor(stroke$Residence_type)
stroke$smoking_status <- factor(stroke$smoking_status)
stroke$stroke <- factor(stroke$stroke)

#changing ever_married variable to binary 1/0 for yes/no
stroke$ever_married[stroke$ever_married == "No"] <- 0
stroke$ever_married[stroke$ever_married == "Yes"] <- 1

stroke$ever_married <- factor(stroke$ever_married)

#changing bmi from chr to numeric
stroke$bmi <- as.numeric(stroke$bmi)
stroke

#standardize all column names to lowercase
names(stroke)[names(stroke) == "Residence_type"] <- "residence_type"

summary(stroke)

head(stroke)
