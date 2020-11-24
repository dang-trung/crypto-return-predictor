#!/usr/bin/env python3
""" __main__ module.

Runs the package when user enters in cmd line: python -m
crypto_return_predictor."""
from crypto_return_predictor import random_forest_predictor

if __name__ == '__main__':
    print(random_forest_predictor.status)
