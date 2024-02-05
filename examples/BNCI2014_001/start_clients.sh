#!/bin/bash

for ((i=1; i<10; i++)); do
    xterm -e "python3 Client.py $i" &
done