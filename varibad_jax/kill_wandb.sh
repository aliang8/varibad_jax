#!/bin/bash

USERNAME=aliang80
PATTERN=wandb-service
pgrep -u $USERNAME -f "^$PATTERN" | while read PID; do
    echo "Killing process ID $PID"
    kill $PID
done