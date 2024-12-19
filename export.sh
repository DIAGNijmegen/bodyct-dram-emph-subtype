#!/usr/bin/env bash

./build.sh

docker save doduo1.umcn.nl/bodyct/releases/dram-emph-subtype:1.0.0 | gzip -c > dram-emph-subtype_1.0.0.tar.gz