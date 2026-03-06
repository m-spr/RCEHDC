#!/usr/bin/env python3
"""Script to train, evaluate, refine online, and re-evaluate the HDC model."""

from hdc import test, online_learning, train


def main():
	train()
	test()
	online_learning()
	test()
	online_learning()
	test()


if __name__ == "__main__":
	main()