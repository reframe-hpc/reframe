/*
 * File:   eatmemory.c
 * Author: Julio Viera <julio.viera@gmail.com>
 *
 * Created on August 27, 2012, 2:23 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <unistd.h>

#if defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGE_SIZE)
#define MEMORY_PERCENTAGE
#endif

#ifdef MEMORY_PERCENTAGE
size_t getTotalSystemMemory(){
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

size_t getFreeSystemMemory(){
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif

bool eat(long total,int chunk){
	long i;
	for(i=0;i<total;i+=chunk){
		short *buffer=malloc(sizeof(char)*chunk);
        if(buffer==NULL){
            return false;
        }
		memset(buffer,0,chunk);
	}
    return true;
}

int main(int argc, char *argv[]){

#ifdef MEMORY_PERCENTAGE
    printf("Currently total memory: %zd\n",getTotalSystemMemory());
    printf("Currently avail memory: %zd\n",getFreeSystemMemory());
#endif

    int i;
    for(i=0;i<argc;i++){
        char *arg=argv[i];
        if(strcmp(arg, "-h")==0 || strcmp(arg,"-?")==0  || argc==1){
            printf("Usage: eatmemory <size>\n");
            printf("Size can be specified in megabytes or gigabytes in the following way:\n");
            printf("#          # Bytes      example: 1024\n");
            printf("#M         # Megabytes  example: 15M\n");
            printf("#G         # Gigabytes  example: 2G\n");
#ifdef MEMORY_PERCENTAGE            
            printf("#%%         # Percent    example: 50%%\n");
#endif            
            printf("\n");
        }else if(i>0){
            int len=strlen(arg);
            char unit=arg[len - 1];
            long size=-1;
            int chunk=1024;
            if(!isdigit(unit) ){
                if(unit=='M' || unit=='G'){
                    arg[len-1]=0;
                    size=atol(arg) * (unit=='M'?1024*1024:1024*1024*1024);
                }
#ifdef MEMORY_PERCENTAGE                
                else if (unit=='%') {
                    size = (atol(arg) * (long)getFreeSystemMemory())/100;
                }
#endif                
                else{
                    printf("Invalid size format\n");
                    exit(0);
                }
            }else{
                size=atoi(arg);
            }
            printf("Eating %ld bytes in chunks of %d...\n",size,chunk);
            if(eat(size,chunk)){
                if(isatty(fileno(stdin))) {
                    printf("Done, press any key to free the memory\n");
                    getchar();
                } else {
                    printf("Done, kill this process to free the memory\n");
                    while(true) {
                        sleep(1);
                    }
                }
            }else{
                printf("ERROR: Could not allocate the memory");
            }
        }
    }

}
