//==============================================================
// https://software.intel.com/sites/default/files/managed/5f/33/roofline_demo_samples.zip
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2018 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================

/*
* This file is intended for use with the "Roofline Analysis in
* Intel(R) Advisor 2017" tutorial video. The tutorial video 
* depicts the use of this code with an Intel(R) Core(TM) i5-6300U  
* processor, compiled with the Intel(R) C++ compiler (version 
* 17.0). Due to the system-dependent nature of Roofline, no 
* guarantee is made that this code will produce the results  
* depicted in the video on your machine (which may have different 
* roofs and/or cache sizes) or with your compiler (which may alter 
* algorithms and therefore arithmetic intensities).
*/
#include <iostream>
#include <random>
#include <chrono>
#include <mpi.h>   
using namespace std;

/*******************************/
/*        Control Panel        */
/*******************************/
//#define GROUP_1
#ifdef GROUP_1
    //#define G1_AOS_SCALAR
    #define G1_SOA_SCALAR
    //#define G1_SOA_VECTOR
#endif /************************/
//#define GROUP_2
#ifdef GROUP_2
    //#define G2_AOS_SCALAR
    //#define G2_SOA_SCALAR
    #define G2_SOA_VECTOR
#endif /************************/
#define GROUP_3
#ifdef GROUP_3
    #define YYYY
    //#define G3_AOS_SCALAR
    //#define G3_SOA_SCALAR
    //#define G3_AOS_VECTOR
    //#define G3_SOA_VECTOR
    //#define G3_SOA_VECTOR_FMAS
#endif /************************/
//#define GROUP_4
#ifdef GROUP_4
    #define G4_SOA
    //#define G4_AOSOA
#endif /************************/

#define MAXVALUE 1000000

#define ARRAY_SIZE_1 1328
//#define REPEAT_1 10000000
//#define REPEAT_1  1000000
//#define REPEAT_1   100000
//#define REPEAT_1   10000
#define REPEAT_1 XXXX

#define ARRAY_SIZE_2 2000
#define REPEAT_2 30000000
#define UNROLL_COUNT 2
#define VECTOR_LENGTH 4
void setupArrays();

// ---- Timer ----
std::chrono::time_point<std::chrono::system_clock> start, stop;
 
//    int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>
//                             (end-start).count();
//    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
// 
//    std::cout << "finished computation at " << std::ctime(&end_time)
//              << "elapsed time: " << elapsed_seconds << "s\n";

//typedef std::chrono::high_resolution_clock Clock;
//typedef std::chrono::time_point<Clock> TimePoint;
//typedef std::chrono::duration<float> Time;
//TimePoint start, stop;

/********** Data Set 1 **********/
// Array of Structures
typedef struct S1_AoS
{
    double a;
    double b;
    double pad1;
    double pad2;
} S1_AoS;
double AoS1_X[ARRAY_SIZE_1];
S1_AoS AoS1_Y[ARRAY_SIZE_1];
// Structure of Arrays
typedef struct S1_SoA
{
    double a[ARRAY_SIZE_1];
    double b[ARRAY_SIZE_1];
    double pad1[ARRAY_SIZE_1];
    double pad2[ARRAY_SIZE_1];
} S1_SoA;
double SoA1_X[ARRAY_SIZE_1];
S1_SoA SoA1_Y;
/********** Data Set 2 **********/
// Structure of Arrays
typedef struct S2_SoA
{
    double a[ARRAY_SIZE_2];
    double b[ARRAY_SIZE_2];
} S2_SoA;
double SoA2_X[ARRAY_SIZE_2];
S2_SoA SoA2_Y;
// Array of Structure of Arrays
typedef struct AoSoA
{
    double a[ARRAY_SIZE_2 / 2];
    double b[ARRAY_SIZE_2 / 2];
} AoSoA;
double AoSoA_X[ARRAY_SIZE_2];
AoSoA AoSoA_Y[2]; 

//int main()
int main(int argc, char *argv[])
{
    MPI::Init(argc, argv);
    setupArrays();
    cout << "Setup complete.\n";

    //############################## Group 1 ##############################//
    // Group 1 is a low arithmetic intensity algorithm intended to display
    // roofline behavior which may initially seem counter-intuitive.
    #ifdef GROUP_1
    cout << "####################### Group 1 #######################\n"
         << "    Algorithm: X = Ya + Yb\n    Data Set 1: " << ARRAY_SIZE_1 
         << " doubles/array.\n";
    #endif
    /******************** AOS - Unvectorized ********************/
    #ifdef G1_AOS_SCALAR
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma novector
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            AoS1_X[i] = AoS1_Y[i].a + AoS1_Y[i].b;
        }
    }
    cout << "Unvectorized AOS loop complete.\n";
    #endif
    /******************** SOA - Unvectorized ********************/
    #ifdef G1_SOA_SCALAR
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma novector
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = SoA1_Y.a[i] + SoA1_Y.b[i];
        }
    }
    cout << "Unvectorized SOA loop complete.\n";
    #endif
    /********************* SOA - Vectorized *********************/
    #ifdef G1_SOA_VECTOR
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = SoA1_Y.a[i] + SoA1_Y.b[i];
        }
    }
    cout << "Vectorized SOA loop complete.\n";
    #endif

    //############################## Group 2 ##############################//
    // Group 2 is not explored in the tutorial video, but it's here if you
    // wish to experiment with it. It has an AI between Groups 1 and 3.
    #ifdef GROUP_2
    cout << "####################### Group 2 #######################\n"
         << "    Algorithm: X = Ya + Yb + Yb\n    Data Set 1: " << ARRAY_SIZE_1
         << " doubles/array.\n";
    #endif
    /******************** AOS - Unvectorized ********************/
    #ifdef G2_AOS_SCALAR
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma novector
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            AoS1_X[i] = AoS1_Y[i].a + AoS1_Y[i].b + AoS1_Y[i].b;
        }
    }
    cout << "Unvectorized AOS loop complete.\n";
    #endif
    /******************** SOA - Unvectorized ********************/
    #ifdef G2_SOA_SCALAR
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma novector
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = SoA1_Y.a[i] + SoA1_Y.b[i] + SoA1_Y.b[i];
        }
    }
    cout << "Unvectorized SOA loop complete.\n";
    #endif
    /********************* SOA - Vectorized *********************/
    #ifdef G2_SOA_VECTOR
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = SoA1_Y.a[i] + SoA1_Y.b[i] + SoA1_Y.b[i];
        }
    }
    cout << "Vectorized SOA loop complete.\n";
    #endif

    //############################## Group 3 ##############################//
    // Group 3 is a high arithmetic intensity algorithm that is intended
    // to demonstrate compute binding and compiler-induced AI changes.
    #ifdef GROUP_3
    cout << "####################### Group 3 #######################\n"
         << "    Algorithm: X = Ya + Ya + Yb + Yb + Yb\n    Data Set 1: " << ARRAY_SIZE_1
         << " doubles/array.\n";
    #endif
    /******************** AOS - Unvectorized ********************/
    #ifdef G3_AOS_SCALAR
    start = std::chrono::system_clock::now();
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma novector
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            AoS1_X[i] = AoS1_Y[i].a + AoS1_Y[i].a + AoS1_Y[i].b + AoS1_Y[i].b + AoS1_Y[i].b;
        }
    }
    cout << "Unvectorized AOS loop complete.\n";
    stop = std::chrono::system_clock::now();
    std::cout << "elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds> (stop-start).count() 
              << "ms\n";
   
    #endif
    /******************** SOA - Unvectorized ********************/
    #ifdef G3_SOA_SCALAR
    start = std::chrono::system_clock::now();
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma novector
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = SoA1_Y.a[i] + SoA1_Y.a[i] + SoA1_Y.b[i] + SoA1_Y.b[i] + SoA1_Y.b[i];
        }
    }
    cout << "Unvectorized SOA loop complete.\n";
    stop = std::chrono::system_clock::now();
    std::cout << "elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds> (stop-start).count() 
              << "ms\n";
    #endif
    /********************* AOS - Vectorized *********************/
    #ifdef G3_AOS_VECTOR
    start = std::chrono::system_clock::now();
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            AoS1_X[i] = AoS1_Y[i].a + AoS1_Y[i].a + AoS1_Y[i].b + AoS1_Y[i].b + AoS1_Y[i].b;
        }
    }
    cout << "Vectorized AOS loop complete.\n";
    stop = std::chrono::system_clock::now();
    std::cout << "elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds> (stop-start).count() 
              << "ms\n";
    #endif
    /********************* SOA - Vectorized *********************/
    #ifdef G3_SOA_VECTOR
    start = std::chrono::system_clock::now();
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = SoA1_Y.a[i] + SoA1_Y.a[i] + SoA1_Y.b[i] + SoA1_Y.b[i] + SoA1_Y.b[i];
        }
    }
    cout << "Vectorized SOA loop complete.\n";
    stop = std::chrono::system_clock::now();
    std::cout << "elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds> (stop-start).count() 
              << "ms\n";
    #endif
    /**************** SOA - Vectorized with FMAs ****************/
    #ifdef G3_SOA_VECTOR_FMAS
    start = std::chrono::system_clock::now();
    for (int r = 0; r < REPEAT_1; r++)
    {
        #pragma unroll (UNROLL_COUNT)
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (int i = 0; i < ARRAY_SIZE_1; i++)
        {
            SoA1_X[i] = (2.0 * SoA1_Y.b[i] + SoA1_Y.b[i]) + SoA1_Y.a[i] * 2.0;
        }
    }
    cout << "Vectorized SOA with FMAs loop complete.\n";
    stop = std::chrono::system_clock::now();
    std::cout << "elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds> (stop-start).count() 
              << "ms\n";
    #endif

    //############################## Group 4 ##############################//
    // Group 4 uses a different data set than the other Groups, and has
    // a medium AI. It is intended to demonstrate cache bandwidth binding.
    #ifdef GROUP_4
    cout << "####################### Group 4 #######################\n"
         << "    Algorithm: X = Ya + Ya + Yb + Yb\n    Data Set 2: " << ARRAY_SIZE_2
         << " doubles/array.\n";
    #endif
    /**************************** SOA ***************************/
    #ifdef G4_SOA
    for (int r = 0; r < REPEAT_2; r++)
    {
        #pragma nounroll
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (int i = 0; i < ARRAY_SIZE_2; i++)
        {
            SoA2_X[i] = SoA2_Y.a[i] + SoA2_Y.a[i] + SoA2_Y.b[i] + SoA2_Y.b[i];
        }
    }
    cout << "SOA loop complete.\n";
    #endif
    /*************************** AOSOA **************************/
    #ifdef G4_AOSOA
    for (int r = 0; r < REPEAT_2; r++)
    {
        for (int j = 0; j < 2; j++)
        {
            #pragma nounroll
            #pragma omp simd simdlen(VECTOR_LENGTH)
            for (int i = 0; i < ARRAY_SIZE_2 / 2; i++)
            {
                AoSoA_X[(j * (ARRAY_SIZE_2 / 2)) + i] = AoSoA_Y[j].a[i] + AoSoA_Y[j].a[i] 
                                                        + AoSoA_Y[j].b[i] + AoSoA_Y[j].b[i];
            }
        }
    }
    cout << "AOSOA loop complete.\n";
    #endif

    MPI::Finalize();
    return EXIT_SUCCESS;
}

void setupArrays()
{
    for (int i = 0; i < ARRAY_SIZE_1; i++)
    {
        SoA1_Y.a[i] = ((rand() % MAXVALUE) + 1) / 3;
        SoA1_Y.b[i] = ((rand() % MAXVALUE) + 1) / 3;
        AoS1_Y[i].a = SoA1_Y.a[i];
        AoS1_Y[i].b = SoA1_Y.b[i];
    }
    for (int i = 0; i < ARRAY_SIZE_2; i++)
    {
        SoA2_Y.a[i] = ((rand() % MAXVALUE) + 1) / 3;
        SoA2_Y.b[i] = ((rand() % MAXVALUE) + 1) / 3;
    }
    for (int i = 0; i < ARRAY_SIZE_2 / 2; i++)
    {
        AoSoA_Y[0].a[i] = SoA2_Y.a[i];
        AoSoA_Y[1].a[i] = SoA2_Y.a[i + (ARRAY_SIZE_2 / 2)];
        AoSoA_Y[0].b[i] = SoA2_Y.b[i];
        AoSoA_Y[1].b[i] = SoA2_Y.b[i + (ARRAY_SIZE_2 / 2)];
    }
}
