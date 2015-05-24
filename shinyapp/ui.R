library(shiny)
library(ggplot2)

shinyUI(fluidPage(

    h1("Who's Gonna Open a Bank Account?"),
    p("This app helps marketing professionals at banks screen prospects for the likelihood that they will open up an account. Enter the parameters below for the prospect that you are considering and the model will make a recommendation on whether to pursue the prospect."),
    p("The",
      a("model", href = "https://github.com/rashanjibowu/bankmarketing/blob/master/bank_marketing_model.md"),
      "that is driving this app was built from a",
      a("bank telemarketing dataset", href = "http://archive.ics.uci.edu/ml/datasets/Bank+Marketing"),
      "available via the UCI Machine Learning Repository."
      ),

    fluidRow(
        column(5,
               plotOutput("plot")
        ),
        column(7, align = "center",
                br(),
                br(),
                br(),
                h4("Recommendation:"),
                htmlOutput("button"),
                br(),
                br(),
                br(),
                p("There is a ",
                    tags$b(textOutput("prob", inline = TRUE)),
                  "chance that this prospect will open an account.")
        )
    ),

    fluidRow(
        column(12,
               hr()
        )
    ),

    fluidRow(
        column(3,
               sliderInput('age',
                           label = 'Age',
                           min = 19,
                           max = 87,
                           value = 21,
                           step = 1
                ),

                selectInput('marital',
                            label = 'Marital Status',
                            choices = list("Single", "Married", "Divorced"),
                            selected = "Single"
                )
        ),
        column(3,
                selectInput('housing_loan',
                            label = 'Has Mortgage?',
                            choices = list("Yes", "No"),
                            selected = "No"
                ),

                selectInput('personal_loan',
                            label = 'Has Personal Loan?',
                            choices = list("Yes", "No"),
                            selected = "No"
                )
        ),
        column(3,
                selectInput('last_contact_type',
                            label = 'Last Mode of Contact',
                            choices = list("Unknown",
                                           "Cellular",
                                           "Telephone")
                ),

                selectInput('previous_outcome',
                            label = 'Previous Outcome',
                            choices = list("Unknown",
                                           "Success",
                                           "Failure",
                                           "Other")
                )
        ),
        column(3,
               sliderInput('contact_count',
                           label = 'Number of Times Contacted',
                           min = 1,
                           max = 50,
                           value = 1,
                           step = 1
               )
        )
    )
))