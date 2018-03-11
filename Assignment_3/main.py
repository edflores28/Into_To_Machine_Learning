import ecoli
import segmentation
import fire
import machine
import neural_network
import argparse

'''
MAIN
'''
# Create a parser for the command line arguments
parser = argparse.ArgumentParser(description="Intro to ML Project 3")
parser.add_argument('-f',action="store_true", default=False, help='Execute forest fires test set')
parser.add_argument('-s',action="store_true", default=False, help='Execute segementation test set')
parser.add_argument('-e',action="store_true", default=False, help='Execute ecoli test set')
parser.add_argument('-m',action="store_true", default=False, help='Execute machines test set')
results = parser.parse_args()

# Perform the tests based on the input
if results.f:
    fire.do_fire()
if results.s:
    segmentation.do_segmentation()
if results.e:
    ecoli.do_ecoli()
if results.m:
    machine.do_machine()
