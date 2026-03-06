#!/usr/bin/env python3
"""Script to train, evaluate, refine online, and re-evaluate the HDC model."""

import sys
from hdc import test, online_learning, online_learning_standard, train


def main():
	# Check if 'onlineHD' argument is passed
	use_standard = len(sys.argv) > 1 and sys.argv[1].lower() == "onlinehd"
	online_fn = online_learning_standard if use_standard else online_learning
	
	train()
	test()
	online_fn()
	test()
	online_fn()
	test()


if __name__ == "__main__":
	main()