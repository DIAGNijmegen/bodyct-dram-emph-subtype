#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t doduo1.umcn.nl/bodyct/releases/dram-emph-subtype:1.0.0 "$SCRIPTPATH"