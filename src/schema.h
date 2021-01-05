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
 * Define structures and functions related to the data's schema (metadata).
 */

#ifndef __SCHEMA__
#define __SCHEMA__

#include <stdio.h>
#include "datatypes.h"

typedef enum {
    UNKNOWN_FORMAT,
    AVATAR_FORMAT,
    EXODUS_FORMAT
} Data_Format;

typedef enum {
    UNKNOWN,
    CONTINUOUS,
    DISCRETE,
    CLASS,
    EXCLUDE
} Attribute_Type;

// This union is the type for storing the imputed value to fill in
// when a value is missing.
// It should probably be called data_value_union.
union data_point_union {
    float Continuous;
    int Discrete;
};
typedef union data_point_union Missing_Value;


/*     // User customization options */
/*     int truth_column; */
/*     int num_skipped_features; */
/*     int *skipped_features; */
/*     int exclude_all_features_above; */
/*     int num_explicitly_skipped_features; */
/*     int *explicitly_skipped_features; */

// Define opaque schema data type and the operations that
// can be performed on it.

typedef struct schema_struct Schema;

/*
 * Read Avatar names file.  All indices are 0-based.
 *
 * Arguments:
 *  namesfile -- file containing schema
 *  manual_target -- Index of target column.  If < 0, 
 *      use the column with type 'class' in the namesfile, or if
 *      there is no class column, default to the last column.
 *  num_manual_exclude -- Number of columns explicitly excluded
 *      (e.g., from command line options).
 *  manual_exclude -- Indices of manually excluded columns.
 */
Schema* read_schema(
    const char* namesfile,
    const Boolean namesfile_is_a_string, // other option is a filename
    int manual_target, 
    int num_manual_exclude, 
    const int* manual_exclude);
void free_schema(Schema* schema);

void write_schema(FILE* fh, const Schema* schema);

uint schema_num_classes(const Schema* schema);
const char* schema_class_name(const Schema* schema);
const char* schema_get_class_value(const Schema* schema, uint valID);
uint schema_class_column(const Schema* schema);

/*
 * Get the number of columns in the input data, including the target
 * column and columns to exclude.
 */
uint schema_num_attr(const Schema* schema);

const char* schema_attr_name(const Schema* schema, uint attrID);

Attribute_Type schema_attr_type(const Schema* schema, uint attrID);

// pre: attrID has type DISCRETE
uint schema_attr_arity(const Schema* schema, uint attrID);
const char* schema_get_discrete_value(const Schema* schema, uint attrID, uint valID);

Missing_Value schema_get_attr_MV(const Schema* schema, uint attrID);
void schema_set_attr_MV(Schema* schema, uint attrID, Missing_Value mv);

/*
 * Check if the attrID column is active.  All columns are active by
 * default except for EXCLUDE columns and those columns manually
 * excluded in the read_schema() call.  Note, in particular, that this
 * include the target column (type CLASS).
 */
Boolean schema_attr_is_active(const Schema* schema, uint attrID);

// void schema_encode(const Schema* schema, uint num_fields, const char** record, float* encoded);
// void schema_decode(const Schema* schema, uint num_fields, const float* record, (const char*)* decoded);


// QUESTIONS:
// * How does Avatar handle / store values from excluded variables?
// * If they are not stored, should we keep a mapping from original column to new attrID?
// * If they are stored, how do we represent those values as floats?


#endif // __SCHEMA__
