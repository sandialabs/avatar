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

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "heartbeat.h"

/*
 * IMPLEMENTATION NOTE: assuming single threaded caller.
 *
 * Internal module storage for remembering previous counter values.
 * Something like this is needed so we can delete the appropriate
 * number of characters from the previous counter display.  
 *
 * This could be adapted to handle multi-threaded callers in a couple
 * of ways.  E.g., synchronizing access to the data structures below
 * to allow only a single caller in at a time, adding a parameter to
 * the functions for holding this bookkeepping, etc.
 */ 
static int _num_counters = 0;
static int* _prev_counters = NULL;
static const int _MINWIDTH = 4;

// Create a buffer for holding the format string.  The format
// string will be ' %wd', with w being the number of digits needed
// to print the counter.  If we are using 64-bit numbers, then the
// width in base 10 will certainly be less than 64 digits.
// Therefore, the number of characters to hold w should be at most 2.
// To be safe, we'll double this and add 4 (for the space, percent
// sign, and letter d).
//static const int FORMAT_BUFFERSIZE = 8+1; // +1 for null terminator
#define FORMAT_BUFFERSIZE 9


// Prototypes for helper functions.
void _write_counters(int num, const int* counters);
void _erase_counters(int num, const int* counters);
int _base_10_width(int n, int min_width);

/*
 * Initialize counter display.
 *
 * Pre: num_counters > 0
 * Post: prints "Building Trees:  0 [ 0 0 ...]" to stdout.
 */
void begin_progress_counters(int num_counters) {
    assert(num_counters > 0);

    // Create local cache for storing previous counter values.
    _num_counters = num_counters;
    _prev_counters = (int*)calloc(_num_counters, sizeof(int));
    if (_prev_counters == NULL) {
        fprintf(stderr, "error: failed to allocate space for _prev_counters\n");
        fflush(stderr);
        exit(-1);
    }

    // Print intial counter state: all at zero.
    printf("Building Trees: ");
    _write_counters(_num_counters, _prev_counters);
    fflush(NULL);
}

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
void update_progress_counters(int num_counters, const int* counters) {
    assert(num_counters == _num_counters);
    assert(counters != NULL);
    int i = 0;

    // Erase previous counter display, and replace with new values.
    _erase_counters(_num_counters, _prev_counters);
    _write_counters(num_counters, counters);
    
    // Store counter values so we know how much to erase at the next update.
    for (i = 0; i < num_counters; ++i) {
        _prev_counters[i] = counters[i];
    }
}

/*
 * Clean up counter display.
 *
 * Pre: begin_progress_counters() called without a matching
 *      end_progress_counters() call.
 * Post: prints end of line to stdout
 */
void end_progress_counters() {
    assert(_prev_counters != NULL); // this indicates a program logic bug
    printf("\n");
    _num_counters = 0;
    free(_prev_counters);
    _prev_counters = NULL;
}

/***** PRIVATE HELPER FUNCTIONS FOR MODULE *****/

// pre: previous counter values have already been erased
void _write_counters(int num, const int* counters) {
    assert(num > 0);
    assert(counters != NULL);

    int i = 0;

    // Create a buffer for holding the format string.  The format
    // string will be ' %wd', with w being the number of digits needed
    // to print the counter.  If we are using 64-bit numbers, then the
    // width in base 10 will certainly be less than 64 digits.
    // Therefore, the number of characters to hold w should be at most 2.
    // To be safe, we'll double this and add 4 (for the space, percent
    // sign, and letter d).
    char format[FORMAT_BUFFERSIZE] = {0};

    // For each counter...
    for (i = 0; i < num; ++i) {
        // ... find number of digits needed to print counter
        int num_digits = _base_10_width(counters[i], _MINWIDTH);
        // ... print counter with leading space, and possibly padded spaces
        int space_needed = snprintf(format, FORMAT_BUFFERSIZE, " %%%dd", num_digits);
        if (space_needed >= FORMAT_BUFFERSIZE || space_needed < 0) {
            fprintf(stderr, "\nError while creating format string in _write_counters()\n");
            fflush(stderr);
            exit(-1);
        }
        printf(format, counters[i]);
    }
    
    // Flush any buffered output.
    fflush(stdout);
}


void _erase_counters(int num, const int* counters) {
    assert(num > 0);
    assert(counters != NULL);

    int i = 0;

    // For each counter ...
    for (i = 0; i < num; ++i) {
        // ... find number of digits needed to print counter
        int num_digits = _base_10_width(counters[i], _MINWIDTH);
        // ... add 1 for leading space
        ++num_digits;
        // ... erase the previously printed count.
        while (num_digits > 0) {
            printf("\b");
            --num_digits;
        }
    }
}

/**
 * Count number of digits needed to represent n in base 10.
 *
 * Parameters
 * ----------
 * n: the non-negative number
 * min_width: lower bound on the number of digits to use
 *
 * Post: returns max(number of digits, min_width).
 */
int _base_10_width(int n, int min_width) {
    assert(n >= 0);
    int num_digits = 1;
    int remainder = n;
    while (remainder >= 10) {
        num_digits += 1;
        remainder /= 10;
    }
    if (num_digits < min_width) {
        num_digits = min_width;
    }
    return num_digits;
}
