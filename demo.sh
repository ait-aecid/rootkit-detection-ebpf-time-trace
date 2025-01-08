python3 evaluate.py -d events -t 0.333 -s 42 -q 9 -r 100 -m offline -g seq
#python3 evaluate.py -d events -t 0.25 -s 42 -q 9 -r 100 -m supervised -g seq
python3 evaluate.py -d events -t 0.333 -s 42 -q 9 -r 1 -m online -g seq
python3 evaluate.py -d events -t 0.333 -s 42 -q 9 -r 100 -m offline -g fun
#python3 evaluate.py -d events -t 0.25 -s 42 -q 9 -r 100 -m supervised -g fun
python3 evaluate.py -d events -t 0.333 -s 42 -q 9 -r 1 -m online -g fun
