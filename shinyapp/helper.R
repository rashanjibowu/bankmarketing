
# Require necessary packages
library(caret)

# load the data
data <- read.csv("data/bank.csv", sep = ";")

# clean the data
data <- plyr::rename(data, c("default" = "in_default",
    "housing" = "housing_loan",
    "loan" = "personal_loan",
    "contact" = "last_contact_type",
    "month" = "last_contact_month",
    "day" = "last_contact_dayofweek",
    "duration" = "last_contact_duration",
    "campaign" = "contact_count",
    "pdays" = "days_since_last_contact",
    "previous" = "prev_campaigns_contact_count",
    "poutcome" = "previous_outcome")
)

# remove data that should have no bearing (time of last contact and duration)
data <- data[, -(10:12)]

# set seed for random calcs
set.seed(123)

# split into training and test data
inTrain <- createDataPartition(y = data$y, p = 0.75, list = FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]

# build new model on training data
glm.revised <- glm(y ~ age +
                       marital +
                       housing_loan +
                       personal_loan +
                       last_contact_type +
                       previous_outcome +
                       contact_count,
                   data = training,
                   family = "binomial")