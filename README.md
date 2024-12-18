# Comparing LLM Metrics
This project has two version of the same file, one for google's gemini api and one with openai's gpt api. These two projects reflect the same principles, simply calling the different backends with respect to the two aforementioned models. The projects use a streamlit dashboard in order to easily view metrics and involve a semantic similarity score as well as ROUGE score, complete with RMSE and total error % calculations.
## Setup Instructions
Add in a .env file with the matching api key name and it should be plug-and-play! (Also make sure to have all dependencies)
## Flow
- Parse dataset
- Build functions to create model responses
- Calculate or parse the responses for metric we are looking for
- Iterate over dataset, store scores in lists for RMSE calculations
- Calculate RMSE and other indicators of success
- Print or store for later use in streamlit
## Timeline
- Reading documentation: 30 mins
- Installation/setup: 45 mins
- Development 150 mins
- BugFixes: 60 mins
## Notes
The gemini project uses waits (time.sleep(60)) to enforce the token limit for student use cases, you can remove these for paid versions as seen with the gpt model use case.
