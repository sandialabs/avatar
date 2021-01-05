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


#ifndef __HEARTBEAT__
#define __HEARTBEAT__

/*
 * Initialize counter display.
 *
 * Pre: num_counters > 0
 * Post: prints "Building Trees:  0 [ 0 0 ...]" to stdout.
 */
void begin_progress_counters(int num_counters);

/*
 * Update the displayed tree counters.  Multiple counters can be shown
 * to display the progress of parallel ensemble learning (e.g., using
 * MPI).
 *
 * Pre: counters != NULL
 * Pre: num_counters > 0 and is the number of counter values stored in
 *      counters.
 * Pre: begin_progress_counters() was called before this
 * Post: The displayed counter values show contents of counters argument.
 */
void update_progress_counters(int num_counters, const int* counters);

/*
 * Clean up counter display.
 *
 * Pre: begin_progress_counters() called without a matching
 *      end_progress_counters() call.
 * Post: prints end of line to stdout
 */
void end_progress_counters(void);

#endif //__HEARTBEAT__
