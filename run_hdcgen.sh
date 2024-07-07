#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# green echo
gecho () {
  echo -e "${GREEN}$1${NC}"
}

# red echo
recho () {
  echo -e "${RED}$1${NC}"
}

export VIVADO_PATH="/Software/xilinx/Vivado"
export VIVADO_VERSION="2022.1"
PROJECT_DIR=$1

if [ -z "$VIVADO_PATH" ];then
  recho "Please set the VIVADO_PATH environment variable to the path to your Vivado installation directory."
fi

if [ -z "$VIVADO_VERSION" ];then
  recho "Please set the VIVADO_VERSION to the version of Vivado to use (e.g. 2020.1)"
fi



python python/make_proj.py $VIVADO_PATH/$VIVADO_VERSION --version $VIVADO_VERSION --pwd $PWD --project_dir "$PROJECT_DIR"