#!/bin/bash

groomer ./runcards/sb3_groomer.json --data results_260503/ww --agent dqn --output results_260503/ww --xrl  --ibmdp-lmut --plot --force &
groomer ./runcards/sb3_groomer.json --data results_260503/ww --agent mdpo --output results_260503/ww --xrl --ibmdp-lmut --plot --force &
groomer ./runcards/sb3_groomer.json --data results_260503/ww --agent ppo --output results_260503/ww --xrl --ibmdp-lmut --plot  --force &
