#!/bin/sh
SessionName=$1
# Use -d to allow the rest of the function to run
bash clean.sh
# python clear_process.py
sleep 2
SessionName='zero'

tmux new-session -d -s $SessionName 'python3 ../service/replay_service.py --size 100000'
tmux new-window -d -n logging 'python3 ../service/log_service.py'

lr=0.0001
# r=0
r=11442
# r=44787

tmux new-window -d -n learn_0
tmux send-keys -t learn_0 'python3 ../agent/si/learner.py --port 2000 --lr '$lr' --seed '$r Enter

tmux new-window -d -n actor_1
tmux send-keys -t actor_1 'python3 ../agent/si/actor.py --aid 1 --port 3000' Enter
# tmux new-window -d -n actor_2
# tmux send-keys -t actor_2 'bash repeated_sac_actor.sh lets-drive 2 4000 '$lr Enter
#tmux new-window -d -n actor_1 'python3 ../agent/si/actor.py --aid 1 --port 3000'
#tmux new-window -d -n actor_2 'python3 ../agent/si/actor.py --aid 2 --port 4000'

tmux attach-session -d -t $SessionName

