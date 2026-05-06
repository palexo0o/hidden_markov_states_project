This is a project which uses markov chains, specifically, a hidden markov model, to model weather state transitions, via the HMMlearn library. It will output various useful data points, such as steady state probabilities, and transition probabilities between each state. We hope to have some degree of accuracy (we do in the end)

The whole project and pipeline is run with the following CLI command run in the main file root of the project:

python main.py 

This will be CLI tool, with the terminal being the source of user input. Arguments are found below:

Valid CLI Arguments/How to Use:

-sd, --startdate : marks the startdate of the weather data you want to use

-ed, --enddate : marks the enddate of the weather data you want to use

-vs, --viewstates : shows the possible weather states, and specifically what weather parameter ranges each state falls under

more will be added soon