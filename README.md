# Rootkit Detection with eBPF Time Tracing

>**Disclaimer**: This repository is the source code release corresponding to academic work,
>if you came here via github or google search it is unlikely that this will be useful for you.
>The documentation assumes familiarity with the corresponding Thesis/Paper.

## Structure

### `probing.py`

````commandline
probing.py [-h] [--iterations ITERATIONS] [--executable EXECUTABLE]
                  [--normal] [--rootkit] [--drop-boundary-events]
                  DESCRIPTION

positional arguments:
  DESCRIPTION           Description of the current experiment, this will be
                        saved in the output's metadata.

options:
  -h, --help            Show this help message and exit.
  --iterations ITERATIONS, -i ITERATIONS
                        Number of times to run the experiment.
                        Default is 100x.
  --executable EXECUTABLE, -e EXECUTABLE
                        Provide an executable for the experiment.
                        Default is 'ls'.
  --normal, -n          Run the normal execution, without anomalies.
  --rootkit, --anormal, -r, -a
                        Run the abnormal execution, with rootkit.
  --drop-boundary-events, -d
                        Drop all events of the first and last PID of each run.
                        These events often miss data. May lead to empty output
                        file if runs <= 2.
  --load, -l [STRESSOR]
                        Put the system under load during the experiment. You can
                        provide a custom executable to do so. Consider shell escaping.
                        Default is 'stress-ng --cpu 10'.
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

Call this with a data file (obtained via `probing.py` / `experiment.sh`) and comment in the desired processing method.

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
  "executable": "ls",
  "iterations": 100,
  "dir_content": ".,..,hide_me_caraxes_asdf,see_me_123,",
  "linux_version": "6.5.0-35-generic",
  "description": [],
  "experiment_begin": "2024-...",
  "experiment_end": "2024-...",
  "events": [...],
  "events_rootkit": [...]
 }
```

## Installation

Install BCC according to your distro, see https://github.com/iovisor/bcc/blob/master/INSTALL.md

On debian that would be: `sudo apt-get install bpfcc-tools linux-headers-$(uname -r)`.

Then fulfill the python dependencies: `pip install -r requirements.txt`.

## Run

> Make sure the rootkit is available in compiled form on the target and the path set in `linux.py`.

On the target machine, execute the experiment:
`python detection.py --normal --rootkit --iterations 100 --executable ./my_custom_ls "My first experiment"`.

This will create a datafile `experimentTimeStamp.json.gz`.

Then, to perform analysis over the data, feed the file to an instance of `process_data.py:Plot`,
or simply run `run.py`.