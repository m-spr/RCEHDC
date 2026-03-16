#!/usr/bin/env python3
"""Script to train, evaluate, refine online, and re-evaluate the HDC model."""

import pathlib
import sys

script_dir = pathlib.Path(__file__).parent.resolve()
repo_root = script_dir.parent
project_dir = repo_root / "mnist_example"

if str(project_dir) not in sys.path:
	sys.path.insert(0, str(project_dir))

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