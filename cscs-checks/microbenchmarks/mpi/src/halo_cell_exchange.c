/* This benchmark emulates a halo cell exchange in n dimensions. The pure
   communication is considered without any stencil computation.*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define NCALLS 10000

int main(int argc, const char *argv[])
{
    MPI_Comm cart_comm, red_comm;
    MPI_Request *request;
    MPI_Status *status;
    FILE *pFile;
    int ndims, reorder, color, end;
    int *dim_size, *periods, *halosize;
    int comm_size, comm_rank, comm_size_cart, comm_rank_cart;
    char *sendbuf, *recvbuf, inputbuf[1000], *pinputbuf;
    double start, stop, deltatmin, deltatmax, elapsed_time;
    int rank_source, rank_dest, i, j;

    if (MPI_Init(NULL, NULL) != 0) {
        fprintf(stderr, "MPI_Init() failed\n");
        exit(1);
    }
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != 0) {
        fprintf(stderr, "MPI_Comm_size() failed\n");
        exit(1);
    }
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != 0) {
        fprintf(stderr, "MPI_Comm_rank() failed\n");
        exit(1);
    }
    if (argc == 1) {
        if (comm_rank == 0) {
            printf("%s inputfile\n", argv[0]);
            printf("ndims dim1 dim2 ... halosize1 halosize2 ...\n");
        }
        exit(0);
    }
    if (comm_rank == 0) {
        if (strcmp(argv[1], "-") == 0) {
            pFile = stdin;
        } else {
            pFile = fopen(argv[1], "r");
        }
    }
    end = 0;
    while (end == 0) {
        if (comm_rank == 0) {
            /* read parameters for every single benchmark line by line */
            end = (fgets(inputbuf, sizeof(inputbuf) - 1, pFile) == NULL);
            if (end == 0) {
                pinputbuf = inputbuf;
                while (*pinputbuf == ' ')
                    pinputbuf++;
                end = !((*pinputbuf >= '0')
                        && (*pinputbuf <= '9'));
                pinputbuf = inputbuf;
            }
        }
        if (MPI_Bcast(&end, 1, MPI_INT, 0, MPI_COMM_WORLD) != 0) {
            fprintf(stderr, "MPI_Bcast() failed\n");
            exit(1);
        }
        if (end == 0) {
            if (comm_rank == 0) {
                /* read number of dimensions */
                sscanf(pinputbuf, "%d", &ndims);
            }
            if (MPI_Bcast(&ndims, 1, MPI_INT, 0, MPI_COMM_WORLD) != 0) {
                fprintf(stderr, "MPI_Bcast() failed\n");
                exit(1);
            }
            dim_size = (int *)malloc(ndims * sizeof(*dim_size));
            periods = (int *)malloc(ndims * sizeof(*periods));
            halosize = (int *)malloc(ndims * sizeof(*halosize));

            for (i = 0; i < ndims; i++) {
                periods[i] = 1;
            }
            reorder = 1;

            j = 1;
            for (i = 0; i < ndims; i++) {
                if (comm_rank == 0) {
                    while (*pinputbuf == ' ')
                        pinputbuf++;
                    while (*pinputbuf != ' ')
                        pinputbuf++;
                    while (*pinputbuf == ' ')
                        pinputbuf++;
                    /* read number of ranks in every dimension */
                    sscanf(pinputbuf, "%d", &dim_size[i]);
                }
                if (MPI_Bcast(&dim_size[i], 1, MPI_INT, 0, MPI_COMM_WORLD) != 0) {
                    fprintf(stderr, "MPI_Bcast() failed\n");
                    exit(1);
                }
                j *= dim_size[i];
            }
            for (i = 0; i < ndims; i++) {
                if (comm_rank == 0) {
                    while (*pinputbuf == ' ')
                        pinputbuf++;
                    while (*pinputbuf != ' ')
                        pinputbuf++;
                    while (*pinputbuf == ' ')
                        pinputbuf++;
                    /* read halo cell size to be communicated in every
                       dimension */
                    sscanf(pinputbuf, "%d", &halosize[i]);
                }
                if (MPI_Bcast(&halosize[i], 1, MPI_INT, 0, MPI_COMM_WORLD) != 0) {
                    fprintf(stderr, "MPI_Bcast() failed\n");
                    exit(1);
                }
            }
            if (j > comm_size) {
                if (comm_rank == 0) {
                    printf("Please run with at least %d MPI ranks.\n", j);
                }
                if (MPI_Finalize() != 0) {
                    fprintf(stderr, "MPI_Finalize() failed\n");
                    exit(1);
                }
                exit(0);
            }

            /* use only the number of ranks required */
            color = (comm_rank < j);
            if (color == 0) {
                if (MPI_Comm_split
                    (MPI_COMM_WORLD, MPI_UNDEFINED, comm_rank,
                     &red_comm) != 0) {
                    fprintf(stderr, "MPI_Comm_split() failed\n");
                    exit(1);
                }
            } else {
                if (MPI_Comm_split(MPI_COMM_WORLD, color, comm_rank, &red_comm)
                    != 0) {
                    fprintf(stderr, "MPI_Comm_split() failed\n");
                    exit(1);
                }
                /* cartesian grid communicator */
                if (MPI_Cart_create
                    (red_comm, ndims, dim_size, periods,
                     reorder, &cart_comm) != 0) {
                    fprintf(stderr, "MPI_Comm_create() failed\n");
                    exit(1);
                }
                if (MPI_Comm_size(cart_comm, &comm_size_cart) != 0) {
                    fprintf(stderr, "MPI_Comm_size() failed\n");
                    exit(1);
                }
                if (MPI_Comm_rank(cart_comm, &comm_rank_cart) != 0) {
                    fprintf(stderr, "MPI_Comm_rank() failed\n");
                    exit(1);
                }

                j = 0;
                for (i = 0; i < ndims; i++) {
                    if (halosize[i] > j) {
                        j = halosize[i];
                    }
                }
                sendbuf = (char *)malloc(ndims * 2 * j * sizeof(char));
                recvbuf = (char *)malloc(ndims * 2 * j * sizeof(char));
                request =
                    (MPI_Request *) malloc(ndims * 2 * 2 * sizeof(MPI_Request));
                status =
                    (MPI_Status *) malloc(ndims * 2 * 2 * sizeof(MPI_Status));

                start = MPI_Wtime();

                for (j = 0; j < NCALLS; j++) {
                    for (i = 0; i < ndims; i++) {
                        /* receive data in every direction */
                        if (MPI_Cart_shift
                            (cart_comm, i, 1, &rank_source, &rank_dest) != 0) {
                            fprintf(stderr, "MPI_Cart_shift() failed\n");
                            exit(1);
                        }
                        if (MPI_Irecv
                            (recvbuf +
                             i * 2 * halosize[i] *
                             sizeof(char), halosize[i],
                             MPI_CHAR, rank_source, 1,
                             cart_comm, request + i * 2) != 0) {
                            fprintf(stderr, "MPI_Irecv() failed\n");
                            exit(1);
                        }
                        if (MPI_Irecv
                            (recvbuf +
                             (i * 2 +
                              1) * halosize[i] *
                             sizeof(char), halosize[i],
                             MPI_CHAR, rank_dest, 1,
                             cart_comm, request + i * 2 + 1) != 0) {
                            fprintf(stderr, "MPI_Irecv() failed\n");
                            exit(1);
                        }
                    }
                    for (i = 0; i < ndims; i++) {
                        /* send data in every direction */
                        if (MPI_Cart_shift
                            (cart_comm, i, 1, &rank_source, &rank_dest) != 0) {
                            fprintf(stderr, "MPI_Cart_shift() failed\n");
                            exit(1);
                        }
                        if (MPI_Isend
                            (sendbuf +
                             i * 2 * halosize[i] *
                             sizeof(char), halosize[i],
                             MPI_CHAR, rank_source, 1,
                             cart_comm, request + i * 2 + ndims * 2) != 0) {
                            fprintf(stderr, "MPI_Irecv() failed\n");
                            exit(1);
                        }
                        if (MPI_Isend
                            (sendbuf +
                             (i * 2 +
                              1) * halosize[i] *
                             sizeof(char), halosize[i],
                             MPI_CHAR, rank_dest, 1,
                             cart_comm, request + i * 2 + 1 + ndims * 2) != 0) {
                            fprintf(stderr, "MPI_Irecv() failed\n");
                            exit(1);
                        }
                    }
                    if (MPI_Waitall(ndims * 2 * 2, request, status) != 0) {
                        fprintf(stderr, "MPI_Waitall() failed\n");
                        exit(1);
                    }
                }

                stop = MPI_Wtime();
                elapsed_time = stop - start;
                if (MPI_Reduce
                    (&elapsed_time, &deltatmin, 1, MPI_DOUBLE,
                     MPI_MIN, 0, cart_comm) != 0) {
                    fprintf(stderr, "MPI_Reduce() failed\n");
                    exit(1);
                }
                if (MPI_Reduce
                    (&elapsed_time, &deltatmax, 1, MPI_DOUBLE,
                     MPI_MAX, 0, cart_comm) != 0) {
                    fprintf(stderr, "MPI_Reduce() failed\n");
                    exit(1);
                }
                if (comm_rank_cart == 0) {
                    printf("halo_cell_exchange %d", comm_size);
                    for (i = 0; i < ndims; i++) {
                        printf(" %d", dim_size[i]);
                    }
                    for (i = 0; i < ndims; i++) {
                        printf(" %d", halosize[i]);
                    }
                    /* print minimum and maximum time per exchange and test */
                    printf(" %e %e\n", deltatmin / NCALLS, deltatmax / NCALLS);
                }
                free(status);
                free(request);
                free(recvbuf);
                free(sendbuf);
                if (MPI_Comm_free(&cart_comm) != 0) {
                    fprintf(stderr, "MPI_Comm_free() failed\n");
                    exit(1);
                }
                if (MPI_Comm_free(&red_comm) != 0) {
                    fprintf(stderr, "MPI_Comm_free() failed\n");
                    exit(1);
                }
            }
            free(halosize);
            free(periods);
            free(dim_size);
        }
    }
    if (MPI_Finalize() != 0) {
        fprintf(stderr, "MPI_Finalize() failed\n");
        exit(1);
    }
}
