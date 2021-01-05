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
/*
 * File:                util.h
 * Authors:             Ken Buch, M. Arthur Munson
 * Company:             Sandia National Laboratories
 * 
 * General purpose utility functions.
 *
 * History:
 *   2006/09    Ken Buch creaated.
 *   2012/3/26  Art Munson moved FClib and CV_ functions to crossval.h
 */

#ifndef __UTIL__
#define __UTIL__

#include "av_utils.h"

typedef struct file_bits_struct {
    char *dirname;
    char *basename;
    char *extension;
} File_Bits;

File_Bits explode_filename(char *filename);
unsigned int factorial(unsigned int n);
void _knuth_shuffle(int size, int *data);
void _opendt_shuffle(int size, int *data, const char* datadir);
int num_digits(int n);

/*
    Reads the next line from the file open as fh
    The read is done in a DOS-friendly way
    The \n and/or \r characters are removed
    read_line returns the number of characters in the line, 0 if EOF, or -1 if error

    NB: caller is responsible for calling free(*str).
 */ 
int read_line(FILE *fh, char **str);

#endif // __UTIL__
