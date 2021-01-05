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
#include <math.h>
#include "av_stats.h"

double av_stats_median_from_sorted_data(const double values[], const int n)
{
  if (n % 2 == 0)
  {
    return (values[n / 2] + values[(n / 2) + 1]) / 2;
  }
  else
  {
    return values[(n - 1) / 2];
  }
}

double av_stats_sd(const double values[], const int n)
{
  /*
    https://en.wikipedia.org/wiki/Standard_deviationm, for the sake of having
    a reference for this.
  */

  // Average
  int i;
  double avg = 0.0;
  double values_copy[n];
  for (i = 0; i < n; i++)
  {
    avg += values[i];
    values_copy[i] = values[i];
  }
  avg /= n;

  // Subtract mean
  for (i = 0; i < n; i++)
  {
    values_copy[i] -= avg;
  }

  // Average mean differences
  double mean_avg = 0.0;
  for (i = 0; i < n; i++)
  {
    mean_avg += values_copy[i];
  }
  mean_avg /= n;
  
  // Square root
  return sqrt(mean_avg);
}

int av_stats_comp_double(const void* d1, const void* d2)
{
  double l = *((double*)d1);
  double r = *((double*)d2);

  if (l > r)
  {
    return 1;
  }

  if (l < r)
  {
    return -1;
  }

  return 0;
}
