import argparse
import time

parser = argparse.ArgumentParser(description='This is the CLI tool for the Hidden Markov Weather Model for modelling weather regimes')

parser.add_argument('-s', '--start', help='this is your start date for your weather data. input as str: YYYY-MM-DD', default='2026-01-01')
parser.add_argument('-e', '--end', help='this is your end date for you weather data. input as a str: YYYY-MM-DD', default='2026-05-01')

args = parser.parse_args()