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
/* Simple C replacements for the GSL random integer functions.
 * 
 * C. Pitts
 * October 2018
 */
#ifndef AV_RNG_H
#define AV_RNG_H
// gsl_rng_uniform_int
/**
 * Generate a pseudorandom integer.
 *
 * @param  max the upper range for the RNG
 * @return pseudorandom unsigned long int
 **/
unsigned long int av_rng_ul_int(unsigned long int max);

// gsl_rng_uniform
/**
 * Generate a pseudorandom double value.
 *
 * @return pseudorandom double value
 **/
double av_rng_uniform();

/**
 * Discrete random variable struct
 */
struct AVDiscRandVar
{
  // Probability distribution
  double* dist;

  // Number of weights in the distribution
  int n;
};

// gsl_ran_discrete_preproc
//unsigned int av_discrete_rand(int n, int* dist);
int av_discrete_rand(struct AVDiscRandVar* var);

/**
 * Stateful adaptation of Park-Miller RNG from Trilinos:
 * https://github.com/trilinos/Trilinos/blob/master/packages/ml/src/Utils/ml_utils.c
 **/
// RNG struct to allocate
struct ParkMiller
{
  int a;
  int m;
  int q;
  int r;
  int state;
};

/**
 * Iterate and produce the next double, adapted from Trilinos source code. Note
 * that calls to this method mutate the ParkMiller struct passed in, so
 * multiple calls to this method *will* return different values. As one would
 * expect from a random number generator, I suppose.
 *
 * @param  rng pointer to ParkMiller struct
 * @return double value between 0 and 1
 **/
double av_pm_iterate(struct ParkMiller* rng);

/**
 * Default initialization (values pulled from Tilinos source code), requires
 * the user to supply a seed value, but all other values are set to what I
 * hope are reasonable defaults.
 *
 * @param  rng ParkMiller struct to initialize
 * @param  seed initial state value
 * @return void
 **/
void av_pm_default_init(struct ParkMiller* rng, int seed);

/**
 * Custom initialization of Park-Miller RNG, the user must supply all values to
 * the method. This is probably more useful in other applications than what we
 * are doing with Avatar.
 *
 * @param  rng ParkMiller struct to initialize
 * @param  a
 * @param  m
 * @param  q
 * @param  r
 * @param  seed the initial state value
 * @return void
 *
 **/
void av_pm_init(struct ParkMiller* rng, int a, int m, int q, int r, int seed);


/**
 * Generate random unsigned long int using a Park-Miller RNG.
 *
 * @param  rng pointer to the RNG
 * @param  n maximum value
 * @return pseudorandom unsigned long int
 **/
unsigned long int av_pm_uniform_ul(struct ParkMiller* rng, unsigned long int n);

/**
 * Generate random int using a Park-Miller RNG.
 *
 * @param  rng pointer to the RNG
 * @param  n maximum value
 * @return pseudorandom unsigned long int
 **/
int av_pm_uniform_int(struct ParkMiller* rng, int n);
#endif // AV_RNG_H
