
source("helper.R")

shinyServer(
    function(input, output) {

        getProbability <- reactive({

            # get model inputs
            inp <- data.frame(age = input$age,
                              marital = tolower(input$marital),
                              housing_loan = tolower(input$housing_loan),
                              personal_loan = tolower(input$personal_loan),
                              last_contact_type = tolower(input$last_contact_type),
                              previous_outcome = tolower(input$previous_outcome),
                              contact_count = input$contact_count)

            # Generate class probabilities first
            classProb <- predict(glm.revised, newdata = inp, type = "response")
            c(classProb, 1 - classProb)
        })

        getClass <- reactive({
            threshold <- 0.10
            classPred <- ifelse(getProbability()[1] < threshold, "no", "yes")
            classPred
        })

        output$button <- renderUI({

            buttonText <- ifelse(getClass() == "yes", "Solid Prospect!", "Don't bother...")

            buttonType <- ifelse(getClass() == "yes", "btn-success", "btn-danger")

            actionButton(NULL,
                         label = buttonText,
                         icon = NULL,
                         class = paste("btn-lg", buttonType)
            )
        })

        output$prob <- renderText({
            paste(round(getProbability()[1] * 100, digits = 1), "%", sep = "")
        })

        output$plot <- renderPlot({
            d <- data.frame(prob = getProbability(), class = c("Yes", "No"))

            ggplot(data = d, aes(x = factor(1), y = prob, fill = factor(class))) +
                geom_bar(stat = "identity", width = 0.5) +
                coord_polar(theta = "y", start = 0) +
                theme(panel.grid.minor = element_blank(),
                      panel.grid.major = element_blank(),
                      legend.title = element_blank(),
                      panel.background = element_blank(),
                      axis.title.x = element_blank(),
                      axis.title.y = element_blank()) +
                scale_fill_brewer(type = "qual", palette=3) +
                labs(title = "Probability of Opening Account")
        })
    }
)