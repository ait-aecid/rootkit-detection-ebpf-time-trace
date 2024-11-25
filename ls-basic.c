#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>

struct linux_dirent {
    long d_ino;
    off_t d_off;
    unsigned short d_reclen;
    char d_name[];
};

#define BUF_SIZE 1024

int main(int argc, char** argv) {
    int fd;
    int nread;
    char buf[BUF_SIZE];
    struct linux_dirent *d;
    int bpos;
    
    char* dir = ".";
    if (argv[1] != NULL){
        dir = argv[1];
    }
    
    // Open the directory
    asm volatile (
        "syscall"
        : "=a" (fd)                    // Output: fd will contain the return value from the syscall
        : "0" (2),                     // Input: syscall number (in RAX)
          "D" (dir),
          "S" (O_RDONLY | O_DIRECTORY) // First argument: pathname (in RDI), second argument: flags (in RSI)
        : "rcx", "r11", "memory"       // Clobbers
    );
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    
    // Perform the getdents syscall using inline assembly
    asm volatile (
        "syscall"
        : "=a" (nread)                    // Output: nread will contain the return value from the syscall
        : "0" (217),                      // Input: syscall number (in RAX)
          "D" (fd),                       // First argument: file descriptor (in RDI)
          "S" (buf),                      // Second argument: buffer (in RSI)
          "d" (BUF_SIZE)                  // Third argument: buffer size (in RDX)
        : "rcx", "r11", "memory"          // Clobbers
    );
    
    if (nread == -1) {
        perror("syscall");
        exit(EXIT_FAILURE);
    }
    
    // Loop over the directory entries
    for (bpos = 0; bpos < nread;) {
        d = (struct linux_dirent *)(buf + bpos);
        printf("%s\n", d->d_name);
        bpos += d->d_reclen;
    }
    
    // Close the directory
    close(fd);
    
    return 0;
}
