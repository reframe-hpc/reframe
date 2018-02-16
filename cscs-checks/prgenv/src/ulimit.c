#include <stdio.h>
#include <sys/resource.h>

int main (int argc, char *argv[])
{
    struct rlimit limit;
    limit.rlim_cur=0;
    limit.rlim_max=0;

    if (getrlimit(RLIMIT_STACK, &limit) == -1) {
        perror("Error");
        return 1;
    }

    if (limit.rlim_cur == RLIM_INFINITY){
        printf("The soft limit is unlimited\n");
    }
    if (limit.rlim_max == RLIM_INFINITY){
        printf("The hard limit is unlimited\n");
    }

  return 0;
}
