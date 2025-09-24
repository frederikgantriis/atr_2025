#!/bin/zsh

PYTHON_FILE="simulator_swarm_robotics_v2.py"

# Each argument set is a string
ARGS_LIST=(
    "disperse"
    "flock"
)

ARGS_LISTT=(
  "5"
  "10"
  "20"
  "30"
  )

# Loop over each argument set
for ARGS in "${ARGS_LIST[@]}"; do
  for ARGSS in "${ARGS_LISTT[@]}"; do
    # Run each set 5 times
    for i in {1..10}; do
        echo "Run $i: python $PYTHON_FILE $ARGS $ARGSS"
        # The unquoted $ARGS splits the string into separate arguments
        python "$PYTHON_FILE" $ARGS $ARGSS
    done
  done
done
