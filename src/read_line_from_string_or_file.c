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
#include "read_line_from_string_or_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int read_line (FILE *fh, char **str);

// skip any initial '\n' or '\r' characters.
static int skip_initial_endlines (const char string[],
				  const int len,
				  const int initial_pos)
{
    if (initial_pos >= len) {
        return len;
    }
    int cur_pos = initial_pos;
    for (; cur_pos < len &&
	   (string[cur_pos] == '\n' ||
	    string[cur_pos] == '\r'); ++cur_pos)
    {}
    return cur_pos;
}

/*
 Reads the next nonempty line from the input string.
 The read is done in a DOS-friendly way.
 The \n and/or \r characters are removed.
 Return the next position to read in the input string.
*/
static int read_line_from_string (char** output_string,
				  int* output_len,
				  const char* input_string,
				  const int input_len,
				  const int input_pos)
{
    int cur_input_pos = skip_initial_endlines (input_string, input_len, input_pos);
    if (cur_input_pos >= input_len || input_string[cur_input_pos] == '\0') {
        *output_string = NULL; // empty line or nothing left
	*output_len = 0;
	return cur_input_pos;
    }
    // cur_input_pos: position of first valid non-endline character.
    //
    // start_input_pos, cur_input_pos: half-exclusive range of valid
    // non-endline characters, that make up the current line.
    const int start_input_pos = cur_input_pos;
    for (; cur_input_pos < input_len; ++cur_input_pos) {
        const char cur_input_char = input_string[cur_input_pos];
	if (cur_input_char == '\n' || cur_input_char == '\r' || cur_input_char == '\0') {
            break; // If no endline, then treat this as the only line.
	}
    }

    const int output_string_len = cur_input_pos - start_input_pos;
    if (output_string_len == 0) {
        *output_string = NULL;
    }
    else {
        char* output_string_copy = malloc ((output_string_len+1) * sizeof (char));
	memcpy (output_string_copy, input_string, output_string_len * sizeof (char));
	output_string_copy[output_string_len] = '\0';
	*output_string = output_string_copy;
    }    
    *output_len = output_string_len;
    return cur_input_pos; // the next spot to read
}

// Like FILE, but for a string.
// Compare to C++ std::istringstream.
typedef struct {
    const char* string;
    int len;
    int pos;
} string_file_handle;

// Read a line from the given string_file_handle, and store the
// resulting pointer in output_string_ptr.  Caller is responsible for
// calling free on it.  Return length of output string.
static int read_line_from_string_handle (string_file_handle* fh,
					 char** output_string_ptr)
{
    if (fh == NULL) {
        return 0;
    }
    else {
        char* output_string = NULL;
	int output_len = 0;
	const int new_input_pos = read_line_from_string (&output_string, &output_len,
							 fh->string, fh->len, fh->pos);
	fh->pos = new_input_pos;
	*output_string_ptr = output_string;
	return output_len;
    }
}

void* open_string_or_file_for_reading (const char* string, const int is_filename)
{
  if (is_filename != 0) {
    FILE* fh = fopen (string, "r");
    return (void*) fh;
  }
  else {
    string_file_handle* sfh = malloc (sizeof (string_file_handle));
    sfh->string = string;
    sfh->len = string == NULL ? 0 : (int) strlen (string);
    sfh->pos = 0;
    return sfh;
  }
}

void close_string_or_file (void* handle, const int is_file)
{
  if (handle != NULL) {
    if (is_file != 0) {
      FILE* fh = (FILE*) handle;
      fclose (fh);
    }
    else {
      string_file_handle* sfh = (string_file_handle*) handle;
      free (sfh);
    }
  }
}

int read_line_from_string_or_file (void* handle, char** buffer, const int is_file)
{
  if (is_file != 0) {
    FILE* fh = (FILE*) handle;
    return read_line (fh, buffer);
  }
  else {
    string_file_handle* sfh = (string_file_handle*) handle;
    return read_line_from_string_handle (sfh, buffer);
  }
}
