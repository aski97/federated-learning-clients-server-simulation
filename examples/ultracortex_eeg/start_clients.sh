#!/bin/bash

for ((i=0; i<6; i++)); do
    xterm -e "python3 Client.py $i" &
done