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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>

#include "util.h"
#include "datatypes.h"
#include "safe_memory.h"

// Prototypes for private module functions.
long _lrand48(const char* folder);

File_Bits explode_filename(char *filename) {
    File_Bits bits;
    char *t;
    unsigned int i;
    t = av_strdup(filename);
    bits.basename = av_strdup(basename(t));
    bits.extension = NULL;
    t = av_strdup(filename);
    bits.dirname = av_strdup(dirname(t));
    int last_dot = -1;
    for (i = 0; i < strlen(bits.basename); i++) {
        if (bits.basename[i] == '.')
            last_dot = i;
    }
    if (last_dot > -1) {
        bits.extension = bits.basename + last_dot+1;
        bits.basename[last_dot] = '\0';
    }
    free(t);
    return(bits);
}

unsigned int factorial(unsigned int n) {
    if (n <= 1)
        return 1;
    return n * factorial(n-1);
}



/*
 *  A one-pass way to random sort an array
 */
void _knuth_shuffle(int size, int *data) {
    int node;
    for (node = size - 1; node >= 0; node--) {
        int rand_num = (int)(lrand48() % (node+1));
        // This doesn't seem to work
        //SWAP(data[node], data[rand]);
        int temp = data[rand_num];
        data[rand_num] = data[node];
        data[node] = temp;
    }
}

/*
 * This is the one-pass way to random sort an array used by OpenDT
 * This version is used for regression tests
 */
void _opendt_shuffle(int size, int *data, const char* datadir) {
    int node;
    for (node = 0; node < size; node++) {
        int rand_num = (int)(_lrand48(datadir) % size);
        int temp = data[rand_num];
        data[rand_num] = data[node];
        data[node] = temp;
    }
}

/*
 * This version of lrand48() reads the random number sequence from a file
 * This version is used in some regression tests where I don't want to or are
 * unable to duplicate the random number sequence of the truth data
 *
 * Pre: folder is the name of a directory user can write to.
 */
long _lrand48(const char* folder) {
    static FILE *fh = NULL;
    const char FNAME[] = "lrand48.opendt";
    char strbuf[1024] = {0};

    if (fh == NULL) {
        // Note that length includes null terminator, b/c it is in FNAME.
        // +1 for the '/' 
        size_t length = strlen(folder) + sizeof(FNAME)/sizeof(FNAME[0]) + 1;
        char* filename = e_calloc(length, sizeof(char));
        int count = snprintf(filename, length, "%s/%s", folder, FNAME);
        assert(count < (int)(length));
        fh = fopen(filename, "r");
        if (fh == NULL) {
            fprintf(stderr, "Failed to open '%s'\n", filename);
            exit(8);
        }
        free(filename);
    }
    fscanf(fh, "%s", strbuf);
    long n = atol(strbuf);
    return n;
}


int num_digits(int n) {
    if (n == 0)
        return 1;
    if (n < 0)
        return (int)log10(-n) + 2;
    return (int)log10(n) + 1;
}


/*
    Reads the next line from the file open as fh
    The read is done in a DOS-friendly way
    The \n and/or \r characters are removed
    read_line returns the number of characters in the line, 0 if EOF, or -1 if error
 */ 
int read_line(FILE *fh, char **str) {
    unsigned int max = 8192;
    char strbuf[max];
    Boolean first_read = TRUE;
    Boolean read_eol = FALSE;
    
    while (read_eol == FALSE) {
        // Read and check for EOF or error
        if (fgets(strbuf, max, fh) == NULL) {
            if (feof(fh) != 0)
                return 0;
            if (ferror(fh) != 0)
                return -1;
        }
        
        //printf("READ: '%s'\n", strbuf);
        // Copy strbuf to *str or append strbuf to *str
        if (first_read == TRUE) {
            first_read = FALSE;
            *str = av_strdup(strbuf);
        } else {
            *str = (char *)realloc(*str, (strlen(*str)+strlen(strbuf)+1)*sizeof(char));
            strcat(*str, strbuf);
        }
        
        // If this read got the end of the line, set read_eol
        if (strbuf[strlen(strbuf)-1] == '\n')
            read_eol = TRUE;
    }
    
    // Return the length of the line with line terminators ...
    int ret_val = strlen(*str);
    // ... but remove the line terminators
    if (strlen(*str) > 0 && ((*str)[strlen(*str)-1] == 10 || (*str)[strlen(*str)-1] == 13))
        (*str)[strlen(*str)-1] = '\0';
    if (strlen(*str) > 0 && ((*str)[strlen(*str)-1] == 10 || (*str)[strlen(*str)-1] == 13))
        (*str)[strlen(*str)-1] = '\0';
    
    return ret_val;
}

