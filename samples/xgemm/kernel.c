/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
# if defined(__APPLE__) && defined(__arm64__)
#include <pthread.h>
# endif

typedef struct gemm_def {
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  double alpha;
  double beta;
  int trans_a;
  int trans_b;
  int aligned_a;
  int aligned_c;
  int prefetch;
  int br_type;
  libxsmm_blasint br_count;
  int br_unroll;
  int tc_config;
} gemm_def;

int g_reps = 0;

LIBXSMM_INLINE void print_help(void) {
  printf("\n\n");
  printf("1. Usage (dense*dense=dense, correctness and performance):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2_BL2viaC, curAL2_BL2viaC\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16, BF1632_FLAT, BF16_FLAT\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("\n\n");
  printf("2. Usage (dense*dense=dense, performance only option available):\n");
  printf("    filename with space-sperated sizes (M N K LDA LDB LDC)\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16, BF1632_FLAT, BF16_FLAT\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    0: no check, otherwise: run check\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("\n\n");
}


LIBXSMM_INLINE
double run_jit_double( const gemm_def*     i_gemm_def,
                       const double*       i_a,
                       const double*       i_b,
                       double*             o_c,
                       const unsigned int  i_print_jit_info) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const double** l_a_addr = (const double**)malloc(i_gemm_def->br_count*sizeof(double*));
  const double** l_b_addr = (const double**)malloc(i_gemm_def->br_count*sizeof(double*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  double l_alpha = i_gemm_def->alpha;
  double l_beta = i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(double);
      if (i_gemm_def->trans_b == 0) {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(double);
      } else {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * sizeof(double);
      }
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->br_type == 0) {
    l_test_jit.dmm = libxsmm_dmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.dmra = libxsmm_dmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.dmra = libxsmm_dmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.dmro = libxsmm_dmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.dmro = libxsmm_dmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->n*sizeof(double),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->k*sizeof(double),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
    } else {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->n*sizeof(double), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->k*sizeof(double), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (const double*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
        }
        l_test_jit.dmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (const double*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
        }
        l_test_jit.dmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_float( const gemm_def*     i_gemm_def,
                      const float*        i_a,
                      const float*        i_b,
                      float*              o_c,
                      const unsigned int  i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const float** l_a_addr = (const float**)malloc(i_gemm_def->br_count*sizeof(float*));
  const float** l_b_addr = (const float**)malloc(i_gemm_def->br_count*sizeof(float*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(float);
      if (i_gemm_def->trans_b == 0) {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(float);
      } else {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * sizeof(float);
      }
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);


  l_start = libxsmm_timer_tick();
  if (i_gemm_def->br_type == 0) {
    l_test_jit.smm = libxsmm_smmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.smra = libxsmm_smmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.smra = libxsmm_smmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.smro = libxsmm_smmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.smro = libxsmm_smmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->n*sizeof(float),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->k*sizeof(float),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
    } else {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->n*sizeof(float), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->k*sizeof(float), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (float*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
        }
        l_test_jit.smra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (float*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
        }
        l_test_jit.smra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_short_int( const gemm_def*     i_gemm_def,
                          const short*        i_a,
                          const short*        i_b,
                          int*                o_c,
                          const unsigned int  i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const short** l_a_addr = (const short**)malloc(i_gemm_def->br_count*sizeof(short*));
  const short** l_b_addr = (const short**)malloc(i_gemm_def->br_count*sizeof(short*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(short);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(short);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.wimm  = libxsmm_wimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.wimm  = libxsmm_wimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.wimm = libxsmm_wimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.wimra = libxsmm_wimmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.wimra = libxsmm_wimmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.wimro = libxsmm_wimmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.wimro = libxsmm_wimmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.wimrs = libxsmm_wimmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(short), i_gemm_def->ldb*i_gemm_def->n*sizeof(short),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.wimrs = libxsmm_wimmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(short), i_gemm_def->ldb*i_gemm_def->n*sizeof(short), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.wimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (short*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (short*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.wimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (short*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (short*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.wimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.wimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_uschar_int( const gemm_def*      i_gemm_def,
                           const unsigned char* i_a,
                           const char*          i_b,
                           int*                 o_c,
                           const unsigned int   i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const unsigned char** l_a_addr = (const unsigned char**)malloc(i_gemm_def->br_count*sizeof(unsigned char*));
  const char** l_b_addr = (const char**)malloc(i_gemm_def->br_count*sizeof(char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(unsigned char);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(char);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.usbimm  = libxsmm_usbimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.usbimm  = libxsmm_usbimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.usbimm = libxsmm_usbimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.usbimra = libxsmm_usbimmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.usbimra = libxsmm_usbimmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.usbimro = libxsmm_usbimmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.usbimro = libxsmm_usbimmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.usbimrs = libxsmm_usbimmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(unsigned char), i_gemm_def->ldb*i_gemm_def->n*sizeof(char),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.usbimrs = libxsmm_usbimmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(unsigned char), i_gemm_def->ldb*i_gemm_def->n*sizeof(char), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.usbimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (unsigned char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.usbimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (unsigned char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.usbimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.usbimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_suchar_int( const gemm_def*      i_gemm_def,
                           const char*          i_a,
                           const unsigned char* i_b,
                           int*                 o_c,
                           const unsigned int   i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const char** l_a_addr = (const char**)malloc(i_gemm_def->br_count*sizeof(char*));
  const unsigned char** l_b_addr = (const unsigned char**)malloc(i_gemm_def->br_count*sizeof(unsigned char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(char);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(unsigned char);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.subimm = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.subimra = libxsmm_subimmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.subimra = libxsmm_subimmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.subimro = libxsmm_subimmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.subimro = libxsmm_subimmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.subimrs = libxsmm_subimmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.subimrs = libxsmm_subimmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.subimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.subimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.subimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.subimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


#if 0
LIBXSMM_INLINE
double run_jit_uschar_uchar( const gemm_def*       i_gemm_def,
                             const unsigned char*  i_a,
                             const char*           i_b,
                             unsigned char*        o_c,
                             const unsigned int    i_print_jit_info ) {
  return 0.0;
}
#endif


LIBXSMM_INLINE
double run_jit_suchar_uchar( const gemm_def*        i_gemm_def,
                             const char*            i_a,
                             const unsigned char*   i_b,
                             unsigned char*         o_c,
                             float                  i_scf,
                             const unsigned int     i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const char** l_a_addr = (const char**)malloc(i_gemm_def->br_count*sizeof(char*));
  const unsigned char** l_b_addr = (const unsigned char**)malloc(i_gemm_def->br_count*sizeof(unsigned char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(char);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(unsigned char);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.sububmm = libxsmm_sububmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.sububmra = libxsmm_sububmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.sububmra = libxsmm_sububmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.sububmro = libxsmm_sububmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.sububmro = libxsmm_sububmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.sububmrs = libxsmm_sububmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.sububmrs = libxsmm_sububmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.subimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmm(i_a, i_b, o_c, &i_scf);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.sububmra(l_a_addr, l_b_addr, o_c, &l_br, &i_scf);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs, &i_scf);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmrs(i_a, i_b, o_c, &l_br, &i_scf);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmm(i_a, i_b, o_c, &i_scf);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.sububmra(l_a_addr, l_b_addr, o_c, &l_br, &i_scf);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs, &i_scf);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmrs(i_a, i_b, o_c, &l_br, &i_scf);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.subimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16_float( const gemm_def*         i_gemm_def,
                               const libxsmm_bfloat16* i_a,
                               const libxsmm_bfloat16* i_b,
                               float*                  o_c,
                               const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }
  if (i_gemm_def->br_type == 0) {
    l_test_jit.bsmm = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16( const gemm_def*         i_gemm_def,
                         const libxsmm_bfloat16* i_a,
                         const libxsmm_bfloat16* i_b,
                               libxsmm_bfloat16* o_c,
                         const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.bmm = libxsmm_bmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}

LIBXSMM_INLINE
double run_jit_bfloat16_float_flat( const gemm_def*         i_gemm_def,
                                    const libxsmm_bfloat16* i_a,
                                    const libxsmm_bfloat16* i_b,
                                    float*                  o_c,
                                    const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }
  if (i_gemm_def->br_type == 0) {
    l_test_jit.bsmm = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16_flat( const gemm_def*         i_gemm_def,
                              const libxsmm_bfloat16* i_a,
                              const libxsmm_bfloat16* i_b,
                                    libxsmm_bfloat16* o_c,
                              const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.bmm = libxsmm_bmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


int main(int argc, char* argv []) {
  char* l_precision = NULL;
  libxsmm_blasint l_lda = 0, l_ldb = 0, l_ldc = 0;
  int l_m = 0, l_n = 0, l_k = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_trans_a = 0;
  int l_trans_b = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_br = 1;
  int l_br_type = 0;
  int l_br_unroll = 0;

  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_matdiff_info l_diff;
  gemm_def l_gemm_def;
  size_t l_i = 0, l_j = 0, l_s = 0, l_t = 0, l_r = 0;
  double l_runtime_c = 0;
  double l_runtime_libxsmm = 0;
  libxsmm_timer_tickint l_start;
  int l_file_input = 0;
  char* l_file_name = NULL;
  FILE *l_file_handle = NULL;
  int l_run_check = 0;

  /* input data */
  double *l_a_d = 0, *l_b_d = 0, *l_c_d = 0;
  float *l_a_f = 0, *l_b_f = 0, *l_c_f = 0;
  short *l_a_w = 0, *l_b_w = 0;
  libxsmm_bfloat16 *l_a_bf = 0, *l_b_bf = 0, *l_c_bf = 0;
  unsigned char *l_ua_b = 0, *l_ub_b;
  char *l_sa_b = 0, *l_sb_b = 0;
  int* l_c_b_i = 0;
  int* l_c_w_i = 0;
  unsigned char* l_c_b_ub = 0;
  float* l_c_bf_f = 0;
  /* Gold data */
  double* l_c_gold_d = 0;
  float* l_c_gold_f = 0;
  libxsmm_bfloat16* l_c_gold_bf = 0;
  int* l_c_gold_w_i = 0;
  int* l_c_gold_b_i = 0;
  unsigned char* l_c_gold_b_ub = 0;
  float* l_c_gold_bf_f = 0;
  double l_total_max_error = 0.0;

  int l_tc_config = 0;

# if defined(__APPLE__) && defined(__arm64__)
#  if 1
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
#  else
  pthread_set_qos_class_self_np( QOS_CLASS_BACKGROUND, 0 );
#  endif
# endif

  /* scaling factor */
  float l_scf = 1.0;

  libxsmm_matdiff_clear(&l_diff);

  /* check argument count for a valid range */
  if ( argc == 20 || argc == 19 ) {
    /* xgemm sizes */
    l_m = atoi(argv[1]);
    l_n = atoi(argv[2]);
    l_k = atoi(argv[3]);
    l_lda = atoi(argv[4]);
    l_ldb = atoi(argv[5]);
    l_ldc = atoi(argv[6]);

    /* some sugar */
    l_alpha = atof(argv[7]);
    l_beta = atof(argv[8]);
    l_aligned_a = atoi(argv[9]);
    l_aligned_c = atoi(argv[10]);
    l_trans_a = atoi(argv[11]);
    l_trans_b = atoi(argv[12]);

    /* arch specific stuff */
    l_precision = argv[14];
    l_br = atoi(argv[16]);
    l_br_unroll = atoi(argv[17]);
    g_reps = atoi(argv[18]);
    if ( argc == 20 ) {
      l_tc_config = atoi(argv[19]);
    } else {
      l_tc_config = 0;
    }

    /* set value of prefetch flag */
    if (strcmp("nopf", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    }
    else if (strcmp("pfsigonly", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_SIGONLY;
    }
    else if (strcmp("BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;
    }
    else if (strcmp("curAL2", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    }
    else if (strcmp("curAL2_BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
    }
    else if (strcmp("AL2", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2;
    }
    else if (strcmp("AL2_BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    if (strcmp("nobr", argv[15]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[15]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[15]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[15]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    l_file_input = 0;
    l_run_check = 1;
  } else if ( argc == 15 || argc == 14 ) {
    l_file_input = 1;
    l_file_name = argv[1];
    l_alpha = atof(argv[2]);
    l_beta = atof(argv[3]);
    l_aligned_a = atoi(argv[4]);
    l_aligned_c = atoi(argv[5]);
    l_trans_a = atoi(argv[6]);
    l_trans_b = atoi(argv[7]);
    l_precision = argv[8];
    l_br = atoi(argv[10]);
    l_br_unroll = atoi(argv[11]);
    if ( argc == 15 ) {
      l_tc_config = atoi(argv[14]);
    } else {
      l_tc_config = 0;
    }

    if (strcmp("nobr", argv[9]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[9]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[9]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[9]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }
    g_reps = atoi(argv[12]);
    l_run_check = atoi(argv[13]);
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  } else {
    print_help();
    return EXIT_FAILURE;
  }

  const char *env_arch = getenv("LIBXSMM_TARGET");
  const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
  int arch_cpuid = libxsmm_cpuid();

  if ((!is_env_SPR && arch_cpuid < LIBXSMM_X86_AVX512_SPR)
       && (l_tc_config)) {
    printf("Warning: external tile configuration will be ingnored\n");
    l_tc_config = 0;
  }

  l_br = (l_br < 1) ? 1 : l_br;
  l_br = (l_br_type == 0) ? 1 : l_br;
  l_br_unroll = (l_br_type == 0) ? 0 : l_br_unroll;

  /* check alpha */
  if ( LIBXSMM_NEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "JIT: alpha needs to be 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "JIT: beta needs to be 0.0 or 1.0!\n");
    exit(EXIT_FAILURE);
  }

 if ( l_file_input != 0 ) {
    l_file_handle = fopen( l_file_name, "r" );
  } else {
    if ( l_trans_b == 0 ) {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
      printf("------------------------------------------------\n");
    } else {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i)^T = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
      printf("------------------------------------------------\n");
    }
  }

  if ((strcmp(l_precision, "DP") == 0) && (l_trans_b == 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_d = (double*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
      l_b_d = (double*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(double), 64);
      l_c_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      l_c_gold_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_a_d[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = libxsmm_rng_f64();
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_b_d[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = libxsmm_rng_f64();
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_d[(l_j * l_ldc) + l_i] = 0.0;
          l_c_gold_d[(l_j * l_ldc) + l_i] = 0.0;
        }
      }

      l_runtime_libxsmm = run_jit_double( &l_gemm_def, l_a_d, l_b_d, l_c_d, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < l_k; l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  l_c_gold_d[(l_j * l_ldc) + l_i] += l_a_d[(l_r * l_lda * l_k) + ((l_s * l_lda) + l_i)] * l_b_d[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_s)];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, l_m, l_n, l_c_gold_d, l_c_d, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_d);
      libxsmm_free(l_b_d);
      libxsmm_free(l_c_d);
      libxsmm_free(l_c_gold_d);
    } while ( l_keep_going );
  }
  else if ((strcmp(l_precision, "DP") == 0) && (l_trans_b != 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_d = (double*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
      l_b_d = (double*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
      l_c_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      l_c_gold_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_a_d[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = libxsmm_rng_f64();
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_b_d[(l_r * l_ldb * l_k) + (l_j * l_ldb) + l_i] = libxsmm_rng_f64();
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_d[(l_j * l_ldc) + l_i] = 0.0;
          l_c_gold_d[(l_j * l_ldc) + l_i] = 0.0;
        }
      }

      l_runtime_libxsmm = run_jit_double( &l_gemm_def, l_a_d, l_b_d, l_c_d, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < l_k; l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  l_c_gold_d[(l_j * l_ldc) + l_i] += l_a_d[(l_r * l_lda * l_k) + (l_s * l_lda) + l_i] *
                                                     l_b_d[(l_r * l_ldb * l_k) + (l_s * l_ldb) + l_j];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, l_m, l_n, l_c_gold_d, l_c_d, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_d);
      libxsmm_free(l_b_d);
      libxsmm_free(l_c_d);
      libxsmm_free(l_c_gold_d);
    } while ( l_keep_going );
  }
  else if ((strcmp(l_precision, "SP") == 0) && (l_trans_b == 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_f = (float*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(float), 64);
      l_b_f = (float*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(float), 64);
      l_c_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_a_f[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = (float)libxsmm_rng_f64();
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_b_f[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = (float)libxsmm_rng_f64();
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_f[(l_j * l_ldc) + l_i] = 0.f;
          l_c_gold_f[(l_j * l_ldc) + l_i] = 0.f;
        }
      }

      l_runtime_libxsmm = run_jit_float( &l_gemm_def, l_a_f, l_b_f, l_c_f, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < l_k; l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  l_c_gold_f[(l_j * l_ldc) + l_i] += l_a_f[(l_r * l_lda * l_k) + (l_s * l_lda) + l_i] *
                                                     l_b_f[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_s];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, l_m, l_n, l_c_gold_f, l_c_f, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_f);
      libxsmm_free(l_b_f);
      libxsmm_free(l_c_f);
      libxsmm_free(l_c_gold_f);
    } while ( l_keep_going );
  }
  else if ((strcmp(l_precision, "SP") == 0) && (l_trans_b != 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_f = (float*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(float), 64);
      l_b_f = (float*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * (size_t)l_br * sizeof(float), 64);
      l_c_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_a_f[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = (float)libxsmm_rng_f64();
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_b_f[(l_r * l_ldb * l_k) + (l_j * l_ldb) + l_i] = (float)libxsmm_rng_f64();
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_f[(l_j * l_ldc) + l_i] = 0.f;
          l_c_gold_f[(l_j * l_ldc) + l_i] = 0.f;
        }
      }

      l_runtime_libxsmm = run_jit_float( &l_gemm_def, l_a_f, l_b_f, l_c_f, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < l_k; l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  l_c_gold_f[(l_j * l_ldc) + l_i] += l_a_f[(l_r * l_lda * l_k) + (l_s * l_lda) + l_i] *
                                                     l_b_f[(l_r * l_ldb * l_k) + (l_s * l_ldb) + l_j];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, l_m, l_n, l_c_gold_f, l_c_f, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_f);
      libxsmm_free(l_b_f);
      libxsmm_free(l_c_f);
      libxsmm_free(l_c_gold_f);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "I16I32") == 0) {
    const int l_k_block = 2;
    double l_max_error = 0;
    int l_k2;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_w = (short*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(short), 64);
      l_b_w = (short*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(short), 64);
      l_c_w_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);
      l_c_gold_w_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_a_w[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = (short)(libxsmm_rng_f64() * 10.0);
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_b_w[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = (short)(libxsmm_rng_f64() * 10.0);
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_w_i[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_w_i[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_short_int( &l_gemm_def, l_a_w, l_b_w, l_c_w_i, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                    l_c_gold_w_i[(l_j * l_ldc) + l_i] += l_a_w[(l_r * l_lda * l_k) + (l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                                         l_b_w[(l_r * l_ldb * l_n) + (l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                  }
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_w_i[(l_j * l_ldc) + l_i] - (double)l_c_w_i[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_w);
      libxsmm_free(l_b_w);
      libxsmm_free(l_c_w_i);
      libxsmm_free(l_c_gold_w_i);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "USI8I32") == 0) {
    const int l_k_block = 4;
    double l_max_error = 0;
    int l_k2;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_ua_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(unsigned char), 64);
      l_sb_b = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(char), 64);
      l_c_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);
      l_c_gold_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_ua_b[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = (unsigned char)(libxsmm_rng_f64() * 5.0);
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_sb_b[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = (char)(libxsmm_rng_f64() * 5.0);
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_b_i[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_b_i[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_uschar_int( &l_gemm_def, l_ua_b, l_sb_b, l_c_b_i, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                    l_c_gold_b_i[(l_j * l_ldc) + l_i] += l_ua_b[(l_r * l_lda * l_k) + (l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                                         l_sb_b[(l_r * l_ldb * l_n) + (l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                  }
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_b_i[(l_j * l_ldc) + l_i] - (double)l_c_b_i[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_ua_b);
      libxsmm_free(l_sb_b);
      libxsmm_free(l_c_b_i);
      libxsmm_free(l_c_gold_b_i);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "SUI8I32") == 0) {
    const int l_k_block = 4;
    double l_max_error = 0;
    int l_k2;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_sa_b = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(char), 64);
      l_ub_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(unsigned char), 64);
      l_c_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);
      l_c_gold_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_sa_b[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = (char)(libxsmm_rng_f64() * 5.0);
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_ub_b[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = (unsigned char)(libxsmm_rng_f64() * 5.0);
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_b_i[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_b_i[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_suchar_int( &l_gemm_def, l_sa_b, l_ub_b, l_c_b_i, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                    l_c_gold_b_i[(l_j * l_ldc) + l_i] += l_sa_b[(l_r * l_lda * l_k) + (l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                                         l_ub_b[(l_r * l_ldb * l_n) + (l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                  }
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_b_i[(l_j * l_ldc) + l_i] - (double)l_c_b_i[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_sa_b);
      libxsmm_free(l_ub_b);
      libxsmm_free(l_c_b_i);
      libxsmm_free(l_c_gold_b_i);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "SUI8UI8") == 0) {
    const int l_k_block = 4;
    double l_max_error = 0;
    int l_k2;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_sa_b = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(char), 64);
      l_ub_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(unsigned char), 64);
      l_c_b_ub = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(unsigned char), 64);
      l_c_gold_b_ub = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(unsigned char), 64);

      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_sa_b[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = (char)(libxsmm_rng_f64() * 2.0);
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_ub_b[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = (unsigned char)(libxsmm_rng_f64() * 2.0);
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_b_ub[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_b_ub[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_suchar_uchar( &l_gemm_def, l_sa_b, l_ub_b, l_c_b_ub, l_scf, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                int tmp = (int)l_c_gold_b_ub[(l_j * l_ldc) + l_i];
                float ftmp;
                for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                  for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                    tmp += l_sa_b[(l_r * l_lda * l_k) + (l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                           l_ub_b[(l_r * l_ldb * l_n) + (l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                  }
                }
                ftmp = (float)tmp;
                ftmp *= l_scf;
                l_c_gold_b_ub[(l_j * l_ldc) + l_i] = (unsigned char)ftmp;
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_b_ub[(l_j * l_ldc) + l_i] - (double)l_c_b_ub[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_sa_b);
      libxsmm_free(l_ub_b);
      libxsmm_free(l_c_b_ub);
      libxsmm_free(l_c_gold_b_ub);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16F32") == 0) {
    const int l_k_block = 2;
    double l_max_error = 0;
    int l_k2;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_c_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
          l_c_gold_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
        }
      }

      l_runtime_libxsmm = run_jit_bfloat16_float( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf_f, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                    union libxsmm_bfloat16_hp tmp_a_f;
                    union libxsmm_bfloat16_hp tmp_b_f;
                    tmp_a_f.i[1] = l_a_bf[(l_r * l_lda * l_k) + (l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2];
                    tmp_a_f.i[0] = 0;
                    tmp_b_f.i[1] = l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                    tmp_b_f.i[0] = 0;
                    l_c_gold_bf_f[(l_j * l_ldc) + l_i] += (float)(tmp_a_f.f * tmp_b_f.f);
                  }
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_bf_f[(l_j * l_ldc) + l_i] - (double)l_c_bf_f[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_bf);
      libxsmm_free(l_b_bf);
      libxsmm_free(l_c_bf_f);
      libxsmm_free(l_c_gold_bf_f);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16") == 0) {
    const int l_k_block = 2;
    double l_max_error = 0;
    int l_k2;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_c_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      l_c_gold_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = 0.0f;
          l_c_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
          l_c_gold_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
        }
      }

      l_runtime_libxsmm = run_jit_bfloat16( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                union libxsmm_bfloat16_hp fprod;
                fprod.i[1] = l_c_gold_bf[(l_j * l_ldc) + l_i];
                fprod.i[0] = 0;
                for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                  for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                    union libxsmm_bfloat16_hp tmp_a_f;
                    union libxsmm_bfloat16_hp tmp_b_f;
                    tmp_a_f.i[1] = l_a_bf[(l_r * l_lda * l_k) + (l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2];
                    tmp_a_f.i[0] = 0;
                    tmp_b_f.i[1] = l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                    tmp_b_f.i[0] = 0;
                    fprod.f += (float)(tmp_a_f.f * tmp_b_f.f);
                  }
                }
                l_c_gold_bf[(l_j * l_ldc) + l_i] = fprod.i[1];
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp_c;
            union libxsmm_bfloat16_hp tmp_gold;
            double l_fabs;

            tmp_c.i[1] = l_c_bf[(l_j * l_ldc) + l_i];
            tmp_c.i[0] = 0;
            tmp_gold.i[1] = l_c_gold_bf[(l_j * l_ldc) + l_i];
            tmp_gold.i[0] = 0;
            l_fabs = fabs((double)tmp_gold.f - (double)tmp_c.f);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_bf);
      libxsmm_free(l_b_bf);
      libxsmm_free(l_c_bf);
      libxsmm_free(l_c_gold_bf);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16F32_FLAT") == 0) {
    double l_max_error = 0;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_c_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
          l_c_gold_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
        }
      }

      l_runtime_libxsmm = run_jit_bfloat16_float_flat( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf_f, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < l_k; l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  union libxsmm_bfloat16_hp tmp_a_f;
                  union libxsmm_bfloat16_hp tmp_b_f;
                  tmp_a_f.i[1] = l_a_bf[(l_r * l_lda * l_k) + (l_s * l_lda) + l_i];
                  tmp_a_f.i[0] = 0;
                  tmp_b_f.i[1] = l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_s];
                  tmp_b_f.i[0] = 0;
                  l_c_gold_bf_f[(l_j * l_ldc) + l_i] += (float)(tmp_a_f.f * tmp_b_f.f);
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_bf_f[(l_j * l_ldc) + l_i] - (double)l_c_bf_f[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_bf);
      libxsmm_free(l_b_bf);
      libxsmm_free(l_c_bf_f);
      libxsmm_free(l_c_gold_bf_f);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16_FLAT") == 0) {
    double l_max_error = 0;
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

      l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
      l_c_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      l_c_gold_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      /* touch A */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch B */
      for (l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_ldb; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = (float)libxsmm_rng_f64();
            l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
          }
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = 0.0f;
          l_c_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
          l_c_gold_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
        }
      }

      l_runtime_libxsmm = run_jit_bfloat16_flat( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                union libxsmm_bfloat16_hp fprod;
                fprod.i[1] = l_c_gold_bf[(l_j * l_ldc) + l_i];
                fprod.i[0] = 0;
                for (l_s = 0; l_s < l_k; l_s++) {
                  union libxsmm_bfloat16_hp tmp_a_f;
                  union libxsmm_bfloat16_hp tmp_b_f;
                  tmp_a_f.i[1] = l_a_bf[(l_r * l_lda * l_k) + (l_s * l_lda) + l_i];
                  tmp_a_f.i[0] = 0;
                  tmp_b_f.i[1] = l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_s];
                  tmp_b_f.i[0] = 0;
                  fprod.f += (float)(tmp_a_f.f * tmp_b_f.f);
                }
                l_c_gold_bf[(l_j * l_ldc) + l_i] = fprod.i[1];
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp_c;
            union libxsmm_bfloat16_hp tmp_gold;
            double l_fabs;

            tmp_c.i[1] = l_c_bf[(l_j * l_ldc) + l_i];
            tmp_c.i[0] = 0;
            tmp_gold.i[1] = l_c_gold_bf[(l_j * l_ldc) + l_i];
            tmp_gold.i[0] = 0;
            l_fabs = fabs((double)tmp_gold.f - (double)tmp_c.f);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_bf);
      libxsmm_free(l_b_bf);
      libxsmm_free(l_c_bf);
      libxsmm_free(l_c_gold_bf);
    } while ( l_keep_going );
  }

  if ( l_file_input != 0 ) {
    fclose( l_file_handle );
  } else {
    printf("------------------------------------------------\n");
  }

  /* Print total max error */
  printf("\n\n Total Max Error %f\n\n", l_total_max_error );

  if ( l_total_max_error >= 0.00005 && l_br_type == 0) {
    return EXIT_FAILURE;
  } else if ( l_total_max_error >= 0.0005 && l_br_type > 0) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

