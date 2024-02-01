#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <# instances>"
    exit 1
fi

instances=$1

for ((i=0; i<instances; i++)); do
    xterm -e "python3 Client.py $i" &
#    python3 "TCPClient.py" " $i" & disown
done