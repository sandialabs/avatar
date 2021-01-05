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
#include "av_utils.h"
#include <string.h>

/*************************************************************************/
char *av_getReturnCodeText(AV_ReturnCode rc) 
{
    switch(rc) {
      case AV_SUCCESS:          return "AV_SUCCESS";
      case AV_ERROR:            return "AV_ERROR";
      case AV_MEMORY_ERROR:     return "AV_MEMORY_ERROR";
      case AV_INPUT_ERROR:      return "AV_INPUT_ERROR";
      case AV_FILE_IO_ERROR:    return "AV_FILE_IO_ERROR";
    }
    return NULL;
}


/*************************************************************************/
AV_ReturnCode _av_lookupBlobInSortedBlobArray(
  AV_SortedBlobArray *sba,  /**< input - sortedIntArray */
  void* blob,               /**< input - blob to lookup */
  int blobCmp(const void* blob1, const void* blob2), /**< input - the
							  compare function */
  int* foundBlob,           /**< output - 1 if find blob, or 0 */
  int* idx                  /**< output - the index if blob is found, or
			       the index the blob would have been at */
){
  int low, high, mid, cmpVal;

  // default returns
  if (foundBlob)
    *foundBlob = -1;
  if (idx)
    *idx = -1;

  // check arguments
  //?FIX could blob be null?
  if (!av_isSortedBlobArrayValid(sba) || !blob || !blobCmp || !foundBlob ||
      !idx) {
    return AV_INPUT_ERROR;
  }

  // early return
  if (sba->numBlob < 1) {
    *foundBlob = 0;
    *idx = 0;
    return AV_SUCCESS;
  }

  // binary search
  low = 0;
  high = sba->numBlob-1;
  while (low <= high) {
    mid = (low+high)/2;
    cmpVal = blobCmp((const void*)blob, (const void*)sba->blobs[mid]);
    if (cmpVal == 0) {
      *foundBlob = 1;
      *idx = mid;
      return AV_SUCCESS;
    } else if (cmpVal < 0) { // blob is before mid
      high = mid-1;
    } else {
      low = mid+1;
    }
  }

  // no match
  *foundBlob = 0;
  *idx = (cmpVal < 0 ? mid : mid+1);
  return AV_SUCCESS;
}


/*************************************************************************/
AV_ReturnCode _av_expandBlobArray(
  int* length,   /**< input/output - length of the array */
  void*** array  /**< input/output - the array */
){
  int new_length;
  void** new_array;

  // test args (note, array doesn't have to be null if length = 0)
  if (!length || *length < 0 || !array) 
    return AV_INPUT_ERROR;

  // do it
  new_length = *length > 0 ? 2*(*length) : 1;
  if (*length == 0)  // malloc in case they didn't pass in NULL
    new_array = malloc(new_length*sizeof(void*));
  else 
    new_array = realloc(*array, new_length*sizeof(void*));
  if (!new_array) {
    av_printfErrorMessage("%s", av_getReturnCodeText(AV_MEMORY_ERROR));
    return AV_MEMORY_ERROR;
  }
 
  // return
  *length = new_length;
  *array = new_array;
  return AV_SUCCESS;
}


/*************************************************************************/
AV_ReturnCode _av_addEntryToSortedBlobArray(
  AV_SortedBlobArray *sba, /**< input/output - sortedBlobArray */
  int idx,                 /**< input - location to add the blob */
  void* blob_p             /**< input - the pointer to the blob */
)
{
  // check args
  if (!av_isSortedBlobArrayValid(sba) || idx < 0 || idx > sba->numBlob ||
      !blob_p) {
    av_printfErrorMessage("%s", av_getReturnCodeText(AV_INPUT_ERROR));
    return AV_INPUT_ERROR;
  }

  // make sure there is room
  if (sba->numBlob >= sba->maxNumBlob)
    _av_expandBlobArray(&sba->maxNumBlob, &sba->blobs);

  // do it
  if (idx < sba->numBlob) {
    memmove((sba->blobs)+idx+1,(sba->blobs)+idx,
	    (sba->numBlob-idx)*sizeof(void*));
  }
  sba->blobs[idx] = blob_p;
  sba->numBlob++;

  return AV_SUCCESS;
}

/*************************************************************************/
AV_ReturnCode av_initSortedBlobArray( AV_SortedBlobArray *sba){ 
  if (sba == NULL){
    av_printfErrorMessage("%s", av_getReturnCodeText(AV_INPUT_ERROR));
    return AV_INPUT_ERROR;
  }

  /*  av_printfLogMessage("Initing sorted blob array");*/

  sba->numBlob = 0;
  sba->maxNumBlob = 0;
  sba->blobs = NULL;
  
  return AV_SUCCESS;
}


/*************************************************************************/
int av_isSortedBlobArrayValid(AV_SortedBlobArray *sba) {
  if (!sba || sba->numBlob < 0 || sba->maxNumBlob < 0 ||
      sba->numBlob > sba->maxNumBlob ||
      (sba->maxNumBlob == 0 && sba->blobs != NULL) ||
      (sba->maxNumBlob > 0 && sba->blobs == NULL)) {
    return 0;
  }
  else
    return 1;
}

/*************************************************************************/
void av_freeSortedBlobArray(AV_SortedBlobArray *sba) {
  if (!av_isSortedBlobArrayValid(sba)){
    return; 
  }

  if (sba->blobs){
    free(sba->blobs);
    sba->numBlob = 0;
    sba->maxNumBlob = 0;
    sba->blobs = NULL;
  }
}

/*************************************************************************/
int av_addBlobToSortedBlobArray(
  AV_SortedBlobArray *sba, /**< input/output - sortedBlobArray */
  void* blob,              /**< input - blob to add */
  int blobCmp(const void*, const void*) /**< input - the blob comparison 
					    function */
){
  AV_ReturnCode rc;
  int foundInt, idx;

  // this call also checks args
  rc = _av_lookupBlobInSortedBlobArray(sba, blob, blobCmp, &foundInt, &idx);
  if (rc < 0)
    return rc;

  // early return
  if (foundInt == 1)
    return 0;

  // do it (this routine allocates memory if needed)
  rc = _av_addEntryToSortedBlobArray(sba, idx, blob);
  if (rc != AV_SUCCESS)
    return rc;
  else
    return 1;
}

/*************************************************************************/
#define AV_VALUE_EQUIV(a,  b, EPS, MIN) \
( ( (a) == (b) ) ||                                                    \
  ( (fabs(a) < (MIN)) && (fabs(b) < (MIN)) ) ||                        \
  ( fabs( ((a)-(b)) / (fabs(a) > fabs(b) ? (a) : (b)) ) < (EPS) )      \
)


/*************************************************************************/

int av_eqf(double x, double y)   {return  AV_VALUE_EQUIV(x, y, FLT_EPSILON, FLT_MIN); }
int av_neqf(double x, double y)  {return !AV_VALUE_EQUIV(x, y, FLT_EPSILON, FLT_MIN);}
int av_ltf(double x, double y)   {if (x < y && !AV_VALUE_EQUIV(x, y, FLT_EPSILON, FLT_MIN)) return 1; else return 0;}
int av_lteqf(double x, double y) {if (x < y ||  AV_VALUE_EQUIV(x, y, FLT_EPSILON, FLT_MIN)) return 1; else return 0;}
int av_gtf(double x, double y)   {if (x > y && !AV_VALUE_EQUIV(x, y, FLT_EPSILON, FLT_MIN)) return 1; else return 0;}
int av_gteqf(double x, double y) {if (x > y ||  AV_VALUE_EQUIV(x, y, FLT_EPSILON, FLT_MIN)) return 1; else return 0;}
int av_eqd(double x, double y)   {return  AV_VALUE_EQUIV(x, y, DBL_EPSILON, DBL_MIN);}
int av_neqd(double x, double y)  {return !AV_VALUE_EQUIV(x, y, DBL_EPSILON, DBL_MIN);}
int av_ltd(double x, double y)   {if (x < y && !AV_VALUE_EQUIV(x, y, DBL_EPSILON, DBL_MIN)) return 1; else return 0;}
int av_lteqd(double x, double y) {if (x < y ||  AV_VALUE_EQUIV(x, y, DBL_EPSILON, DBL_MIN)) return 1; else return 0;}
int av_gtd(double x, double y)   {if (x > y && !AV_VALUE_EQUIV(x, y, DBL_EPSILON, DBL_MIN)) return 1; else return 0;}
int av_gteqd(double x, double y) {if (x > y ||  AV_VALUE_EQUIV(x, y, DBL_EPSILON, DBL_MIN)) return 1; else return 0;}


char* av_strdup(const char* str)
{
  size_t len = strlen(str);
  char* copy = malloc(len + 1);
  memcpy(copy, str, len + 1);
  return copy;
}
