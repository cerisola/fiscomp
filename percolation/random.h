/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

/*
 * This code is derived from the official basic C PCG implementation (see:
 * https://github.com/imneme/pcg-c-basic). It has been slightly modified to
 * remove unnecessary functions and defines and to add two extra functions
 * srand_pcg and rand_pcg and the macro RAND_MAX_PCG to allow using it as
 * similarly as possible to the stdlib random functions.
 * -- Federico Cerisola
*/

#ifndef PCG_BASIC_H_INCLUDED
#define PCG_BASIC_H_INCLUDED 1

#include <inttypes.h>

struct pcg_state_setseq_64 {    /* Internals are *Private*. */
    uint64_t state;             /* RNG state.  All values are possible. */
    uint64_t inc;               /* Controls which RNG sequence (stream) is */
                                /* selected. Must *always* be odd. */
};
typedef struct pcg_state_setseq_64 pcg32_random_t;

/* If you *must* statically initialize it, here's one. */

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

/* pcg32_srandom(initstate, initseq)
 * pcg32_srandom_r(rng, initstate, initseq):
 *     Seed the rng.  Specified in two parts, state initializer and a
 *     sequence selection constant (a.k.a. stream id) */

void pcg32_srandom(uint64_t initstate, uint64_t initseq);
void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate,
                     uint64_t initseq);

/* pcg32_random()
 * pcg32_random_r(rng)
 *     Generate a uniformly distributed 32-bit random number */

uint32_t pcg32_random(void);
uint32_t pcg32_random_r(pcg32_random_t* rng);

/* drop in replacement functions for stdlib */
#define RAND_MAX_PCG 4294967295

void srand_pcg(uint64_t seed);
uint32_t rand_pcg();

#endif /* PCG_BASIC_H_INCLUDED */