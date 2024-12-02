import subprocess
import sys

ROOTKIT_NAME = "caraxes"
KERNEL_OBJECT_PATH = "/home/ubuntu/caraxes/"


def list_modules() -> [str]:
    modules = []
    result = subprocess.run(["lsmod"], capture_output=True, text=True)
    output = result.stdout
    first = True
    for line in output.split("\n"):
        if first:
            first = False
            continue
        module = line.split(" ", maxsplit=1)[0]
        if module:
            modules.append(module)
    return modules


def insert_rootkit() -> None:
    result = subprocess.run(["sudo", "insmod", KERNEL_OBJECT_PATH + ROOTKIT_NAME + ".ko"], capture_output=True, text=True)
    exitcode = result.returncode
    if exitcode != 0:
        raise Exception(result.stderr)


def remove_rootkit() -> None:
    result = subprocess.run(["sudo", "rmmod", ROOTKIT_NAME], capture_output=True, text=True)
    exitcode = result.returncode
    if exitcode != 0:
        raise Exception(result.stderr)


def shell(cmd: str) -> str:
    command = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = command.communicate()
    command.wait()
    stdout_str = stdout.decode('utf-8')
    print(stdout_str, file=sys.stderr, end='')
    return stdout_str


def run_background(cmd: str) -> subprocess.Popen:
    process = subprocess.Popen(cmd.split(" "))
    return  process
