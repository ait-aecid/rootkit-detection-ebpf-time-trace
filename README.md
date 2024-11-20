# Rootkit Detection with eBPF Time Tracing

>**Disclaimer**: This repository is the source code release corresponding to academic work,
>if you came here via github or google search it is unlikely that this will be useful for you.
>The documentation assumes familiarity with the corresponding Thesis/Paper.

## Structure

### `detection.py`

````commandline
usage: detection.py [-h] [--iterations ITERATIONS] [--executable EXECUTABLE]
                    [--normal] [--rootkit]
                    ...

positional arguments:
  description           Description of the current experiment, this will be
                        saved in the output's metadata.

options:
  -h, --help            Show this help message and exit.
  --iterations ITERATIONS, -i ITERATIONS
                        Number of times to run the experiment.
                        Default is 100x.
  --executable EXECUTABLE, -e EXECUTABLE
                        Provide an executable for the experiment.
                        Default is `ls`.
  --normal, -n          Run the normal execution, without anomalies.
  --rootkit, --anormal, -r, -a
                        Run the anormal execution, with rootkit.
````

This is the heart of the project that performs the experiment.

At the top of the file, specify functions for which probes will be inserted at their enter and return points.
Edit as needed.

Define the structure of the experiment directory here.
Edit as needed.

The script goes on to insert the eBPF probes via the help of BCC.
Then it runs the executable `i` times, while collecting the timestamps of the probes.
After that it loads the rootkit, runs the executable again i times and collects the stamps.
Lastly it saves the data into gzip compressed json file.

For example the invocation of `stress` also happens here, but is commented out.

### `experiment.sh`

Shell file for running an experiment by providing a remote target to conduct the experiment on, ssh key access is assumed.
Result file will be copied to the local working directory and a run of data processing will be conducted.

* `-e` Specify executable, e.g. `/usr/bin/ls`

* `-i` Specify number of iterations

The user at the target of the experiment has to have root rights,
since insertion of eBPF probes and loading of the rootkit requires so.

### `run.py`

Call this with a data file (obtained via `user.py` / `experiment.sh`) and comment in the desired processing method.

### `process_data.py:Plot`

Feed a data file to this to construct the in-memory representation from points to intervals and conduct all sorts of representations.

### `kernel.c`

The eBPF probe code.

### `linux.py`

Helper functions.

Define the path where the rootkit is saved on the target system.

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
