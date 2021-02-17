/*

Copyright (c) 2019 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/nsimd.h>

// Required for random generation
#include "./benches.hpp"

// Google benchmark
#include <benchmark/benchmark.h>

// std
#include <cmath>

// Sleef
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <sleef.h>
#pragma GCC diagnostic pop

// MIPP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wzero-length-array"
#endif
#include <mipp.h>
#pragma GCC diagnostic pop

// -------------------------------------------------------------------------

static const int sz = 1024;

template <typename Random> static f64 *make_data(int sz, Random r) {
  f64 *data = (f64 *)nsimd_aligned_alloc(sz * 8);

  for (int i = 0; i < sz; ++i) {
    data[i] = r();
  }
  return data;
}

static f64 *make_data(int sz) {
  f64 *data = (f64 *)nsimd_aligned_alloc(sz * 8);

  for (int i = 0; i < sz; ++i) {
    data[i] = f64(0);
  }
  return data;
}

f64 rand_param0() {
  return nsimd::benches::rand<f64>(-std::numeric_limits<f64>::infinity(),
                                   std::numeric_limits<f64>::infinity());
}

f64 rand_param1() {
  return nsimd::benches::rand<f64>(-std::numeric_limits<f64>::infinity(),
                                   std::numeric_limits<f64>::infinity());
}

// -------------------------------------------------------------------------

extern "C" {
void __asm_marker__nsimd_cpu_mul() {}
}

void nsimd_cpu_mul(benchmark::State &state, f64 *_r, f64 *_0, f64 *_1, int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / vlen(f64)) * vlen(f64);
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__nsimd_cpu_mul");
      // code: nsimd_cpu_mul
      int n = vlen(f64);

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        nsimd::storea(_r + i,
                      nsimd::mul(nsimd::loada(_0 + i, f64(), nsimd::cpu()),
                                 nsimd::loada(_1 + i, f64(), nsimd::cpu()),
                                 f64(), nsimd::cpu()),
                      f64(), nsimd::cpu());
      }
      // code: nsimd_cpu_mul
      __asm__ __volatile__("callq __asm_marker__nsimd_cpu_mul");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(nsimd_cpu_mul, f64, make_data(sz),
                  make_data(sz, &rand_param0), make_data(sz, &rand_param1), sz);

extern "C" {
void __asm_marker__nsimd_avx2_mul() {}
}

void nsimd_avx2_mul(benchmark::State &state, f64 *_r, f64 *_0, f64 *_1,
                    int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / vlen(f64)) * vlen(f64);
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul");
      // code: nsimd_avx2_mul
      int n = vlen(f64);

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        nsimd::storea(_r + i,
                      nsimd::mul(nsimd::loada(_0 + i, f64()),
                                 nsimd::loada(_1 + i, f64()), f64()),
                      f64());
      }
      // code: nsimd_avx2_mul
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(nsimd_avx2_mul, f64, make_data(sz),
                  make_data(sz, &rand_param0), make_data(sz, &rand_param1), sz);

extern "C" {
void __asm_marker__nsimd_avx2_mul_unroll2() {}
}

void nsimd_avx2_mul_unroll2(benchmark::State &state, f64 *_r, f64 *_0, f64 *_1,
                            int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / nsimd::len(nsimd::pack<f64, 2>())) *
       nsimd::len(nsimd::pack<f64, 2>());
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul_unroll2");
      // code: nsimd_avx2_mul_unroll2
      int n = nsimd::len(nsimd::pack<f64, 2>());

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        nsimd::storea(_r + i,
                      nsimd::mul(nsimd::loada<nsimd::pack<f64, 2> >(_0 + i),
                                 nsimd::loada<nsimd::pack<f64, 2> >(_1 + i)));
      }
      // code: nsimd_avx2_mul_unroll2
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul_unroll2");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(nsimd_avx2_mul_unroll2, f64, make_data(sz),
                  make_data(sz, &rand_param0), make_data(sz, &rand_param1), sz);

extern "C" {
void __asm_marker__nsimd_avx2_mul_unroll3() {}
}

void nsimd_avx2_mul_unroll3(benchmark::State &state, f64 *_r, f64 *_0, f64 *_1,
                            int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / nsimd::len(nsimd::pack<f64, 3>())) *
       nsimd::len(nsimd::pack<f64, 3>());
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul_unroll3");
      // code: nsimd_avx2_mul_unroll3
      int n = nsimd::len(nsimd::pack<f64, 3>());

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        nsimd::storea(_r + i,
                      nsimd::mul(nsimd::loada<nsimd::pack<f64, 3> >(_0 + i),
                                 nsimd::loada<nsimd::pack<f64, 3> >(_1 + i)));
      }
      // code: nsimd_avx2_mul_unroll3
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul_unroll3");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(nsimd_avx2_mul_unroll3, f64, make_data(sz),
                  make_data(sz, &rand_param0), make_data(sz, &rand_param1), sz);

extern "C" {
void __asm_marker__nsimd_avx2_mul_unroll4() {}
}

void nsimd_avx2_mul_unroll4(benchmark::State &state, f64 *_r, f64 *_0, f64 *_1,
                            int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / nsimd::len(nsimd::pack<f64, 4>())) *
       nsimd::len(nsimd::pack<f64, 4>());
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul_unroll4");
      // code: nsimd_avx2_mul_unroll4
      int n = nsimd::len(nsimd::pack<f64, 4>());

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        nsimd::storea(_r + i,
                      nsimd::mul(nsimd::loada<nsimd::pack<f64, 4> >(_0 + i),
                                 nsimd::loada<nsimd::pack<f64, 4> >(_1 + i)));
      }
      // code: nsimd_avx2_mul_unroll4
      __asm__ __volatile__("callq __asm_marker__nsimd_avx2_mul_unroll4");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(nsimd_avx2_mul_unroll4, f64, make_data(sz),
                  make_data(sz, &rand_param0), make_data(sz, &rand_param1), sz);

extern "C" {
void __asm_marker__std_mul() {}
}

void std_mul(benchmark::State &state, f64 *_r, volatile f64 *_0,
             volatile f64 *_1, int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / 1) * 1;
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__std_mul");
      // code: std_mul
      int n = 1;

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        *(_r + i) = *(_0 + i) * *(_1 + i);
      }
      // code: std_mul
      __asm__ __volatile__("callq __asm_marker__std_mul");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(std_mul, f64, make_data(sz), make_data(sz, &rand_param0),
                  make_data(sz, &rand_param1), sz);

extern "C" {
void __asm_marker__MIPP_mul() {}
}

void MIPP_mul(benchmark::State &state, f64 *_r, f64 *_0, f64 *_1, int sz) {
  // Normalize size depending on the step so that we're not going out of
  // boundaies (Required when the size is'nt a multiple of `n`, like for
  // unrolling benches)
  sz = (sz / vlen(f64)) * vlen(f64);
  try {
    for (auto _ : state) {
      __asm__ __volatile__("callq __asm_marker__MIPP_mul");
      // code: MIPP_mul
      int n = vlen(f64);

#pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {
        mipp::store(_r + i, mipp::mul<f64>(mipp::load<f64>(_0 + i),
                                           mipp::load<f64>(_1 + i)));
      }
      // code: MIPP_mul
      __asm__ __volatile__("callq __asm_marker__MIPP_mul");
    }
  } catch (std::exception const &e) {
    std::string message("ERROR: ");
    message += e.what();
    state.SkipWithError(message.c_str());
  }
}
BENCHMARK_CAPTURE(MIPP_mul, f64, make_data(sz), make_data(sz, &rand_param0),
                  make_data(sz, &rand_param1), sz);

BENCHMARK_MAIN();

