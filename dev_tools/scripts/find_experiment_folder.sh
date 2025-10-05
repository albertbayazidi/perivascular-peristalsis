#!/bin/bash

parameter=$1
value=$2

grep -P --color=always -H "\"$parameter\"\s*:\s*\[[^\]]*\b\K$value\b" results/comparison/**/**/*.json


