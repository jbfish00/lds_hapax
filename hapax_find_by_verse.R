library(scriptuRs)
library(dplyr)
library(tidytext)
library(tidyr)
library(spacyr)
library(readr)




BOM <- read_csv("BOM.csv")


# Calculate the majority of 'written' status for each event
event_written_majority <- BOM %>%
  group_by(event) %>%
  summarize(mean_written = mean(written)) %>%
  mutate(event_written = ifelse(mean_written > 0.5, 1, 0))

# Join the calculated majority back to the original dataframe
BOM <- BOM %>%
  left_join(event_written_majority, by = "event")

# Select and rename the relevant columns
BOM <- BOM %>%
  select(-mean_written)


write.csv(BOM, "BOM.csv")
