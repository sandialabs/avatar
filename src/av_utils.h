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
/* Simple C replacements for fclib
 * 
 */
#ifndef AV_UTILS_H
#define AV_UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#ifdef HAVE_AVATAR_FCLIB
#include "fc.h"
#else
//these types appear in structs and function parameters,
//but aren't actually used in the non-fclib code paths
typedef int FC_Dataset;
typedef int FC_Mesh;
typedef int FC_Variable;
typedef int FC_AssociationType;
#endif

typedef enum {
  AV_SUCCESS = 0,
  AV_ERROR = -1,
  AV_MEMORY_ERROR = -2,
  AV_INPUT_ERROR = -3,
  AV_FILE_IO_ERROR = -4
} AV_ReturnCode;


char *av_getReturnCodeText(AV_ReturnCode rc);

#define av_exitIfError(code) { \
  if( (AV_ReturnCode)(code) != AV_SUCCESS ) { \
  fprintf(stderr,"Exiting with error %s (%d) at [%s:%d]\n",av_getReturnCodeText(code),(code),__FILE__,__LINE__); fflush(NULL); exit(code); \
  } \
}

#define av_exitIfErrorPrintf(code, message, ...) \
  { \
    if ( (AV_ReturnCode)(code) != AV_SUCCESS ) { \
      fprintf(stderr, "Exiting with error %s (%d) at [%s:%d]: " message "\n", \
              av_getReturnCodeText(code), (code), __FILE__, __LINE__, ##__VA_ARGS__); \
      fflush(NULL); \
      exit(code); \
    } \
  }

#define av_printfErrorMessage(message, ...)  \
  { \
      fprintf(stderr, "AV Error:%s[%s:%d]: " message "\n", \
              __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
      fflush(NULL); \
  }

#define av_missingFCLIB() \
  fprintf(stderr, "%s:%d: To use this feature, place the " \
      "fclib 1.6.1 source tree in avatar/util and rebuild.", __FILE__, __LINE__); \
  exit(-1);

typedef struct av_sorted_blob_array {
  int numBlob;     /**< number of blobs */
  int maxNumBlob;  /**< total allocated space (max possible number of blobs) */
  void** blobs;    /**< array of blobs */
} AV_SortedBlobArray;


// sorted blob array routines
AV_ReturnCode av_initSortedBlobArray(AV_SortedBlobArray *sba);
int av_isSortedBlobArrayValid(AV_SortedBlobArray* sba);
void av_freeSortedBlobArray(AV_SortedBlobArray *sba);
int av_addBlobToSortedBlobArray(AV_SortedBlobArray *sba, void* blob,
		   int blobCmp(const void* blob1, const void* blob2));

// floating point comparisons
int av_eqf(double x, double y);
int av_neqf(double x, double y);
int av_ltf(double x, double y);
int av_lteqf(double x, double y);
int av_gtf(double x, double y);
int av_gteqf(double x, double y);
int av_eqd(double x, double y);
int av_neqd(double x, double y);
int av_ltd(double x, double y);
int av_lteqd(double x, double y);
int av_gtd(double x, double y);
int av_gteqd(double x, double y);

// implementation of non-standard C function strdup
char* av_strdup(const char* str);

#endif
