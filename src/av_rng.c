/********************************************************************************** 
Avatar Tools 
Copyright (c) 2019, National Technology and Engineering Solutions of Sandia, LLC
All rights reserved. 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer  in the
  documentation and/or other materials provided with the distribution.


3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For questions, comments or contributions contact 
Philip Kegelmeyer, wpk@sandia.gov 
*******************************************************************************/
#include <stdlib.h>
#include <limits.h>
#include "av_rng.h"

double av_pm_iterate(struct ParkMiller* rng)
{
  int hi   = rng->state / rng->q;
  int lo   = rng->state % rng->q;
  int test = rng->a * lo - rng->r * hi;

  if (test > 0) rng->state = test;
  else rng->state = test + rng->m;

  double rand_num = (double) rng->state / (double) rng->m;
  return rand_num;
}

void av_pm_default_init(struct ParkMiller* rng, int seed)
{
  // Pulled from Trilinos implementation:
  //int    a = 16807, m = 2147483647, q = 127773, r = 2836;
  rng->a = 16807;
  rng->m = 2147483647;
  rng->q = 127773;
  rng->r = 2836;
  rng->state = seed;
}

void av_pm_init(struct ParkMiller* rng, int a, int m, int q, int r, int seed)
{
  rng->a = a;
  rng->m = m;
  rng->q = q;
  rng->r = r;
  rng->state = seed;
}

unsigned long int av_pm_uniform_ul(struct ParkMiller* rng, unsigned long int n)
{
  double p = av_pm_iterate(rng);
  return (unsigned long int)(p * ULONG_MAX) % n;
}

int av_pm_uniform_int(struct ParkMiller* rng, int n)
{
  double p = av_pm_iterate(rng);
  return (int)(p * INT_MAX) % n;
}

unsigned long int av_rng_ul_int(unsigned long int max)
{
  return random() % max;
}

double av_rng_uniform()
{
  return random() / RAND_MAX + 1.0;
}

int av_discrete_rand(struct AVDiscRandVar* var)
{
  // Summation array
  int i;
  double summation[var->n];
  summation[0] = var->dist[0];

  for (i = 1; i < var->n; i++)
  {
    summation[i] = var->dist[i] + summation[i - 1];
  }

  // Generate random double value
  double val = (double)rand()/(double)(RAND_MAX);

  // Simple order N search
  int index = 0;
  while (summation[index] < val)
  {
    index += 1;
  }

  return index;
}


/**
 * TODO: add random discrete sample struct/functions
 * reset weights could be a no-op, compare pointers
 */
