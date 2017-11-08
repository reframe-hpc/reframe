#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CUDA_HOME
#   define CUDA_HOME ""
#endif

int main() {
    char *cuda_home_compile = CUDA_HOME;
    char *cuda_home_runtime = getenv("CUDA_HOME");
    if (cuda_home_runtime &&
        strnlen(cuda_home_runtime, 256) &&
        strnlen(cuda_home_compile, 256) &&
        !strncmp(cuda_home_compile, cuda_home_runtime, 256)) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE\n");
        printf("Compiled with CUDA_HOME=%s, ran with CUDA_HOME=%s\n",
               cuda_home_compile,
               cuda_home_runtime ? cuda_home_runtime : "<null>");
    }

    return 0;
}
