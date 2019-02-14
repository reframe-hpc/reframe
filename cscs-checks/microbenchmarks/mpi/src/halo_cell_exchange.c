#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, const char* argv[]){
    MPI_Comm cart_comm, red_comm;
    MPI_Request *request;
    MPI_Status *status;
    int ndims, reorder, color, end, ierr;
    int *dim_size, *periods, *halosize;
    int comm_size, comm_rank, comm_size_cart, comm_rank_cart;
    char *sendbuf, *recvbuf, inputbuf[1000], *pinputbuf;
    double start, stop, deltatmin, deltatmax, ttt;
    int rank_source, rank_dest, i, j;

    if (argc>1){
        printf("%s < ndims dim1 dim2 ... halosize1 halosize2 ...\n", argv[0]);
        exit(0);
    }
    ierr = MPI_Init(NULL, NULL);
    if (ierr!=0) exit(1);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (ierr!=0) exit(1);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (ierr!=0) exit(1);
    end = 0;
    while (end == 0){
        if (comm_rank==0){
            end = (fgets(inputbuf, sizeof(inputbuf), stdin) == NULL);
            if (end == 0){
                pinputbuf = inputbuf;
                while (pinputbuf[0]==' ') pinputbuf++;
                end = !((pinputbuf[0]>='0')&&(pinputbuf[0]<='9'));
                pinputbuf = inputbuf;
            }
        }
        MPI_Bcast(&end, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (end == 0){
            if (comm_rank==0){
                sscanf(pinputbuf, "%d", &ndims);
            }
            MPI_Bcast(&ndims, 1, MPI_INT, 0, MPI_COMM_WORLD);
            dim_size = (int*) malloc(ndims*sizeof(int));
            periods = (int*) malloc(ndims*sizeof(int));
            halosize = (int*) malloc(ndims*sizeof(int));

            for (i=0; i<ndims; i++){
                periods[i] = 1;
            }
            reorder = 1;

            j = 1;
            for (i=0; i<ndims; i++){
                if (comm_rank==0){
                    while (pinputbuf[0]==' ') pinputbuf++;
                    while (pinputbuf[0]!=' ') pinputbuf++;
                    while (pinputbuf[0]==' ') pinputbuf++;
                    sscanf(pinputbuf, "%d", &dim_size[i]);
                }
                MPI_Bcast(&dim_size[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
                j *= dim_size[i];
            }
            for (i=0; i<ndims; i++){
                if (comm_rank==0){
                    while (pinputbuf[0]==' ') pinputbuf++;
                    while (pinputbuf[0]!=' ') pinputbuf++;
                    while (pinputbuf[0]==' ') pinputbuf++;
                    sscanf(pinputbuf, "%d", &halosize[i]);
                }
                MPI_Bcast(&halosize[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
            if (j>comm_size){
                if (comm_rank==0){
                    printf("Please run with at least %d MPI ranks.\n", j);
                }
                ierr = MPI_Finalize();
                if (ierr!=0) exit(1);
                exit(0);
            }

            color = (comm_rank < j);
            if (color == 0){
                ierr = MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, comm_rank, &red_comm);
            }else{
                ierr = MPI_Comm_split(MPI_COMM_WORLD, color, comm_rank, &red_comm);
                ierr = MPI_Cart_create(red_comm, ndims, dim_size, periods, reorder, &cart_comm);
                if (ierr!=0) exit(1);
                ierr = MPI_Comm_size(cart_comm, &comm_size_cart);
                if (ierr!=0) exit(1);
                ierr = MPI_Comm_rank(cart_comm, &comm_rank_cart);
                if (ierr!=0) exit(1);

                j=0;
                for (i=0; i<ndims; i++){
                    if (halosize[i]>j){
                        j=halosize[i];
                    }
                }
                sendbuf = (char*) malloc(ndims*2*j*sizeof(char));
                recvbuf = (char*) malloc(ndims*2*j*sizeof(char));
                request = (MPI_Request*) malloc(ndims*2*2*sizeof(MPI_Request));
                status = (MPI_Status*) malloc(ndims*2*2*sizeof(MPI_Status));

                start = MPI_Wtime ();

                for (j=0; j<10000; j++){
                    for (i=0; i<ndims; i++){
                        ierr = MPI_Cart_shift(cart_comm, i, 1, &rank_source, &rank_dest);
                        if (ierr!=0) exit(1);
                        ierr = MPI_Irecv(recvbuf+i*2*halosize[i]*sizeof(char), halosize[i], MPI_CHAR, rank_source, 1, cart_comm, request+i*2);
                        if (ierr!=0) exit(1);
                        ierr = MPI_Irecv(recvbuf+(i*2+1)*halosize[i]*sizeof(char), halosize[i], MPI_CHAR, rank_dest, 1, cart_comm, request+i*2+1);
                        if (ierr!=0) exit(1);
                    }
                    for (i=0; i<ndims; i++){
                        ierr = MPI_Cart_shift(cart_comm, i, 1, &rank_source, &rank_dest);
                        if (ierr!=0) exit(1);
                        ierr = MPI_Isend(sendbuf+i*2*halosize[i]*sizeof(char), halosize[i], MPI_CHAR, rank_source, 1, cart_comm, request+i*2+ndims*2);
                        if (ierr!=0) exit(1);
                        ierr = MPI_Isend(sendbuf+(i*2+1)*halosize[i]*sizeof(char), halosize[i], MPI_CHAR, rank_dest, 1, cart_comm, request+i*2+1+ndims*2);
                        if (ierr!=0) exit(1);
                    }
                    ierr = MPI_Waitall(ndims*2*2, request, status);
                    if (ierr!=0) exit(1);
                }

                stop = MPI_Wtime ();
                ttt = stop - start;
                ierr = MPI_Reduce (&ttt, &deltatmin, 1, MPI_DOUBLE, MPI_MIN, 0, cart_comm);
                if (ierr!=0) exit(1);
                ierr = MPI_Reduce (&ttt, &deltatmax, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
                if (ierr!=0) exit(1);
                if (comm_rank_cart == 0){
                    printf ("halo_cell_exchange %d", comm_size);
                    for (i=0; i<ndims; i++){
                        printf (" %d", dim_size[i]);
                    }
                    for (i=0; i<ndims; i++){
                        printf (" %d", halosize[i]);
                    }
                    printf (" %e %e\n", deltatmin, deltatmax);
                }
                free(status);
                free(request);
                free(recvbuf);
                free(sendbuf);
                ierr = MPI_Comm_free(&cart_comm);
                if (ierr!=0) exit(1);
                ierr = MPI_Comm_free(&red_comm);
                if (ierr!=0) exit(1);
            }
        }
    }
    ierr = MPI_Finalize();
}
