# Rootkit Detection with eBPF Time Tracing

This repository contains code to collect time measurements of kernel functions that are manipulated by rootkits when they hide files, as well as a semi-supervised detection method that analyzes shifts of kernel function execution times. This implementation relies on the open-source rootkit [CARAXES](https://github.com/ait-aecid/caraxes), which wraps the filldir function to manipulate the outcomes of file enumerations, e.g., when executing the `ls` command. Time measurements are taken from several functions in the getdents system call (including filldir) using eBPF probes. For detection we apply a simple machine learning model based on statistical testing. We refer to the following publication for detailed explanations of data collection and anomaly detection mechanisms. If you use any of the resources provided in this repository, please cite the following publication:
* Landauer, M., Alton, L., Lindorfer, M., Skopik, F., Wurzenberger, M., & Hotwagner, W. (2025). Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity. Under Review.

## Rootkit and eBPF probing

The following steps setup up the rootkit and explain how to collect time measurements from kernel functions. In case that you are only interested in the detection of anomalies and want to use our public data sets, you can skip this section.

### Setup

The rootkit and probing has been tested on Linux kernels 5.15-6.11 and Python 3.10. To run the tools, download this repository and install the following dependencies required to run the rootkit and probing mechanism.

```sh
ubuntu@ubuntu:~$ git clone https://github.com/ait-aecid/rootkit-detection-ebpf-time-trace.git
ubuntu@ubuntu:~$ cd rootkit-detection-ebpf-time-trace
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ sudo apt update
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ sudo apt install python3-bpfcc make gcc flex bison linux-headers-$(uname -r)
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ pip install -r requirements.txt
```

Some scenarios require additional resources. Specifically, the ls-basic scenario requires to compile the ls-basic script and the system load scenario requires to install stress-ng. If you do not want to use these scenarios, you can skip the following dependencies.

```sh
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ gcc -o ls-basic ls-basic.c
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ sudo apt install stress-ng
```

The following commands are mandatory, because the rootkit manipulates the getdents system call by default, however, only manipulation of filldir is currently supported for probing. Download the rootkit, replace `hooks.h` with the file provided in this repository (this ensures that filldir is hooked rather than getdents), and compile the rootkit.

```sh
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ cd ..
ubuntu@ubuntu:~$ git clone https://github.com/ait-aecid/caraxes.git
ubuntu@ubuntu:~$ cd caraxes/
ubuntu@ubuntu:~/caraxes$ cp ../rootkit-detection-ebpf-time-trace/hooks.h .
ubuntu@ubuntu:~/caraxes$ sudo make
```

If you have troubles installing the rootkit or want to test if it works as expected, check out the ReadMe on the [CARAXES](https://github.com/ait-aecid/caraxes) github page.

Afterwards, return to this repository and open `linux.py` to edit the variable `KERNEL_OBJECT_PATH` so that it points to the caraxes folder that you just cloned; the default path is `"/home/ubuntu/caraxes/"`.

```sh
ubuntu@ubuntu:~/caraxes$ cd ../rootkit-detection-ebpf-time-trace
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ vim linux.py
```

### Measurement of kernel function timings

Now you are ready to run the probing mechanism that will automatically inject probes into the kernel, start the rootkit, store time measurement data into a file, and stop the rootkit. To trigger system calls, the script thereby creates a directory with files to be hidden and executes `ls` 100 times (can be modified with the `-i` flag) while polling the probes. The script allows to collect measurments with the rootkit (`--rootkit` flag), without the rootkit (`--normal` flag), or both, and supports several scenarios. Run the default scenario with the following command:

```sh
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ sudo python3 probing.py --normal --rootkit
compiling eBPF probes...
probes compiled!

attached bpf probes:
iterate_dir-enter
iterate_dir-return
dcache_readdir-enter
dcache_readdir-return
filldir64-enter
filldir64-return
verify_dirent_name-enter
verify_dirent_name-return
touch_atime-enter
touch_atime-return

Running experiment with ls for 100 times.
Iteration 0...
detection_PID: 70133
Iteration 1...
detection_PID: 70134
...
Iteration 99...
detection_PID: 70338
polled 40 times!
done with the "rootkit version"
Experiment finished, saving output.
Saved data to events/events_2025-01-17T09:59:52.183801_rootkit.json.gz
412K    events/events_2025-01-17T09:59:52.183801_rootkit.json.gz
```

Measurements are stored in the `events` directory. Check out the help page with `python3 probing.py -h` to learn about other parameters that allow to set up data collection in other scenarios (e.g., using `ls-basic` instead of `ls` or simulating system load) and assign names to different runs (`--description`). Check out `repeat_seq.sh` for some parameterized commands; in fact, we used this script to collect our public data sets. Note that we only consider some of the available kernel functions from the getdents system call. To specify which functions the probing mechanism should attach probes to, open `probing.py` and add or remove function names in the `probe_points` list in the beginning of the file. 

## Anomaly detection

Running the anomaly detection algorithm only requires to install the python dependencies. If you have not done so already in the previous step, run the following command to install the requirements with pip.

```sh
ubuntu@ubuntu:~$ git clone https://github.com/ait-aecid/rootkit-detection-ebpf-time-trace.git
ubuntu@ubuntu:~$ cd rootkit-detection-ebpf-time-trace
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ pip install -r requirements.txt
```

Then, download and extract the data set we provide on [Zenodo](https://zenodo.org/records/14679675). If you generated your own data in the previous step and want to use it, skip this step.

```sh
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ wget https://zenodo.org/records/14679675/files/events.zip
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ unzip events.zip
```

Now you are ready to run the anomaly detection as follows. Specify the directory containing measurement data (`-d`), the fration of normal data used for training (`-t`), the mode of operation (`-m`) and the grouping function (`-g`). The script will load all files from the specified directory, split them into training and testing data (summarized in the output), compute and print detection metrics, and plot a confusion matrix.

```sh
ubuntu@ubuntu:~/rootkit-detection-ebpf-time-trace$ python3 evaluate.py -d events -t 0.333 -m offline -g fun
100%|█████████████████████████████████████████████| 1250/1250 [02:47<00:00,  7.45it/s]
Processed all files from events

Normal batches: 750
  Normal batches for training: 250
    default: 50
    file_count: 50
    system_load: 50
    ls_basic: 50
    filename_length: 50
  Normal batches for testing: 500
    default: 100
    file_count: 100
    system_load: 100
    ls_basic: 100
    filename_length: 100
Anomalous batches: 500
  default: 100
  file_count: 100
  system_load: 100
  ls_basic: 100
  filename_length: 100

Results (Run 1)
 Threshold=3.5111917342151415e-16
 Time=0.0027740001678466797
 TP=499
 FP=9
 TN=491
 FN=1
 TPR=R=0.998
 FPR=0.018
 TNR=0.982
 P=0.9822834645669292
 F1=0.9900793650793651
 ACC=0.99
 MCC=0.9801254640896192

Confusion Matrix (Run 1)
Predicted
default     file_count  system_load ls_basic    filename_length
Pos   Neg   Pos   Neg   Pos   Neg   Pos   Neg   Pos   Neg
100   0     100   0     100   0     100   0     100   0      Pos - Actual default
3     97    100   0     99    1     100   0     2     98     Neg - Actual default
100   0     100   0     100   0     100   0     100   0      Pos - Actual file_count
100   0     0     100   100   0     100   0     100   0      Neg - Actual file_count
100   0     100   0     100   0     100   0     100   0      Pos - Actual system_load
100   0     100   0     4     96    100   0     100   0      Neg - Actual system_load
100   0     100   0     100   0     99    1     100   0      Pos - Actual ls_basic
100   0     100   0     100   0     2     98    100   0      Neg - Actual ls_basic
99    1     100   0     100   0     100   0     100   0      Pos - Actual filename_length
3     97    100   0     99    1     100   0     0     100    Neg - Actual filename_length
```

Check out the manual using `python3 evaluate.py -h` to learn more about available parameters of this script. Also, have a look at `demo.sh` to see the parameterized commands that we used for the evaluation in our paper.

# Citation

If you use any of the resources provided in this repository, please cite the following publication:
* Landauer, M., Alton, L., Lindorfer, M., Skopik, F., Wurzenberger, M., & Hotwagner, W. (2025). Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity. Under Review.

Then, to perform analysis over the data, feed the file to an instance of `process_data.py:Plot`,
or simply run `run.py`.
