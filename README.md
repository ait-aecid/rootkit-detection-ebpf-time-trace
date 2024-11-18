# Rootkit Detection with eBPF Time Tracing

>**Disclaimer**: This repository is the source code release corresponding to academic work,
>if you came here via github or google search it is unlikely that this will be useful for you.
>The documentation assumes familiarity with the corresponding Thesis/Paper.

## Structure

### `experiment.sh`

Shell file for running an experiment by providing a remote target to conduct the experiment on, ssh key acces is assumed.
Result file will be copied to the local working directory and a run of data procesing will be conducted.

* `-e` Specify executable, e.g. `/usr/bin/ls`

* `-i` Specify number of iterations

The user at the target of the experiment has to have root rights,
since insertion of eBPF probes and loading of the rootkit requires so.

### `run.py`

Call this with a data file (obtained via `user.py` / `experiment.sh`) and comment in the wanted processing.

### `process_data.py:Plot`

Feed a data file to this to construct the in memory representation from points to intervals and conduct all sorts of representations.

### `user.py`

Heart of the project.
This performs the experiment.

At the top of the file specify functions at who's enter and return points the probes shall be inserted.

Define the structure of the experiment directory here.

The script goes on to insert the eBPF probes via the help of BCC.
Then it runs the executable i times, while collecting the timestamps of the probes.
After that it loads the rootkit, runs the executable again itimes and collects the stamps.
Lastly it saves the data into gzip compressed json file.

### `kernel.c`

The eBPF probe code.

### `linux.py`

Helper functions.

Define the path of the rootkit, where it is saved on the target system here.

### `data_classes.py`

Python definition of the json data structure of the experiment.

```
{
  "executable": "./getpid_opendir_readdir_root",
  "iterations": 10,
  "linux_version": "6.5.0-35-generic",
  "events": [...],
  "events_rootkit": [...]
}
```