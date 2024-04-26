#!/bin/bash

USERNAME=anthony
PATTERN=wandb-service
pgrep -u $USERNAME -f "^$PATTERN" | while read PID; do
    echo "Killing process ID $PID"
    kill $PID
done