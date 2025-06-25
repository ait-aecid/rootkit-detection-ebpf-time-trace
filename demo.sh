python3 evaluate.py -d events -t 0.333 -s 42 -r 100 -m offline -g fun -a shift -q 9
python3 evaluate.py -d events -t 0.333 -s 42 -r 100 -m offline -g seq -a shift -q 9
python3 evaluate.py -d events -t 0.333 -s 42 -r 100 -m offline -g fun -a ann
python3 evaluate.py -d events -t 0.333 -s 42 -r 100 -m offline -g seq -a ann
python3 evaluate.py -d events -t 0.333 -s 42 -r 100 -m offline -g fun -a svm
python3 evaluate.py -d events -t 0.333 -s 42 -r 100 -m offline -g seq -a svm
python3 evaluate.py -d events -t 0.333 -s 42 -r 1 -m online -g fun -a shift -q 9
python3 evaluate.py -d events -t 0.333 -s 42 -r 1 -m online -g seq -a shift -q 9
