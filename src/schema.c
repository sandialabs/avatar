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
#include <ctype.h> // this can be removed later, probably
#include <math.h>   // this can be removed later, probably
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "schema.h"
#include "array.h"
#include "datatypes.h"
#include "read_line_from_string_or_file.h"
#include "safe_memory.h"
#include "util.h"

typedef struct variable_struct {
    char* name;
    Attribute_Type type;
    uint arity;
    char** domain;
    Missing_Value imputed_value;
    Boolean active;
} VariableInfo;


// Declare prototypes for helper functions.
void _clear_variable_info(uint num_vars, VariableInfo* vi);
Boolean _parse_attribute(char* buffer, VariableInfo* result);
Boolean _set_target_info(Schema* schema, int manual_target);
char** _tokenize(const char* s, const char* delim, uint* num_tokens);





////////////////////////



// This is typedef'd to Schema in header file.
struct schema_struct {
    uint targetID;
    uint num_classes;
    uint* num_examples_per_class;
    uint num_attributes;
    VariableInfo* attributes;
    
    // TODO: do we need something like these?
    // int num_fclib_seq;
    // int* global_offset;
    // Exo_Data exo_data;
}; // Schema;


/************** PUBLIC API ****************/

Schema* 
read_schema(
    const char* namesfile,
    const Boolean namesfile_is_a_string,
    int manual_target,
    int num_manual_exclude,
    const int* manual_exclude)
{
    assert(namesfile != NULL);
    const int namesfile_is_a_file = (namesfile_is_a_string == TRUE) ? 0 : 1;
    
    Schema* schema = (Schema*)e_calloc(1, sizeof(Schema));
    Boolean success = TRUE;
    FILE* fh = NULL;
    uint i = 0;

    // This loop exists so we can break out if there is an error.
    char* buffer = NULL;
    do {
	if (namesfile_is_a_file){
	    fh = fopen(namesfile, "r");
	}
	else {
	  fh = fmemopen((char*)namesfile, strlen(namesfile), "r");	
	}
        if (fh == NULL) {
            fprintf(stderr, "Failed to open .names file: '%s'\n", namesfile);
            success = FALSE;
            break;
        }

        // Make initial guess for *upper* bound on number of
        // variables.  We'll remove any excess space later.
        uint max_attr = 3; // 100;
        schema->attributes = (VariableInfo*)e_calloc(max_attr, sizeof(VariableInfo));

        // For each line in the names file...
        free(buffer); buffer=NULL;
        uint line_num = 0;
        uint column_count = 0;
        while (success && read_line(fh, &buffer) > 0) {
            line_num++;
            strip_lt_whitespace(buffer);

            // Skip empty lines and comment lines.
            if (*buffer == '\0' || *buffer == '#') {
                free(buffer); buffer=NULL;
                continue;
            }

            // Parse the line into attribute info.
            Boolean line_specifies_attribute = NULL != strchr(buffer, ':');
            if (line_specifies_attribute) {
                // Make sure we have enough space to hold the attribute.
                if (schema->num_attributes >= max_attr) {
                    // Grow to double capacity.  Doubling results in
                    // expected linear time.
                    max_attr *= 2;
                    schema->attributes = (VariableInfo*)e_realloc(
                        schema->attributes, max_attr*sizeof(VariableInfo));
                    // Zero initialize the new entries in array.
                    uint num_filled = schema->num_attributes;
                    VariableInfo* new_entries = schema->attributes;
                    new_entries += num_filled;
                    memset(new_entries, 0, (max_attr - num_filled)*sizeof(VariableInfo));
                }

                Boolean okay = _parse_attribute(buffer, &(schema->attributes[column_count]));
                if (okay) {
                    ++column_count;
                    schema->num_attributes += 1;
                }
                else {
                    success = FALSE;
                }
            }
            else {
                fprintf(stderr, "warning: ignoring line %d with unexpected format in %s\n", line_num, namesfile);
            }

            free(buffer); buffer=NULL;
        } // end reading lines from names file
        fclose(fh);
	fh = NULL;
        if (!success) { break; }
        
        // Trim excess space from attributes.
        //schema->attributes = (VariableInfo*)e_realloc(
           // schema->attributes, schema->num_attributes*sizeof(VariableInfo));


        // Malloc more memory for buffer (free'd above and never realloc'd)
        buffer = (char*)malloc(128*sizeof(char));

        // Fill in names for nameless attributes.
        char format[128] = {0};
        sprintf(format, "Atr%%0%dd", num_digits(schema->num_attributes));
        for (i = 0; i < schema->num_attributes; ++i) {
            VariableInfo* attr = &(schema->attributes[i]);
            // If name is empty, create one.
            if (attr->name[0] == '\0') {
                char name_buffer[128] = {0};
                size_t buffsize = sizeof(name_buffer) / sizeof(name_buffer[0]);
                int num_written = snprintf(buffer, buffsize, format, i);
                if (num_written >= (int)buffsize) {
                    // name_buffer is too small; this should never happen...
                    fprintf(stderr, "error: buffer too small at %s, line %d\n", __FILE__, __LINE__);
                    success = FALSE;
                    break;
                }
                free(attr->name);
                attr->name = e_strdup(name_buffer);
            }
        }
        if (!success) { break; }

        // Exclude the attributes specified manually. (I.e., on command line.)
        // Final exclusion set is union of attributes excluded in
        // names file and those excluded manually.
        int curr;
        for (curr = 0; curr < num_manual_exclude; ++curr) {
            Boolean validID = manual_exclude[curr] >= 0 
                && manual_exclude[curr] < (int)(schema->num_attributes);
            if (!validID) {
                fprintf(
                    stderr, 
                    "error: cannot exclude column %d b/c it is outside valid range [1,%d]\n", 
                    manual_exclude[curr] + 1,
                    schema->num_attributes);
                success = FALSE;
                break;
            }
            schema->attributes[ manual_exclude[curr] ].active = FALSE;
        }
        if (!success) { break; }
        
        success = _set_target_info(schema, manual_target);
        if (!success) { break; }

        // If we get here, everything worked.
    } while (FALSE);
    free(buffer); buffer=NULL;

    if (fh != NULL) {
        fclose(fh);
        fh = NULL;
    }
    
    // If anything in the read failed, cleanup.
    if (!success) {
        free_schema(schema);  
        schema = NULL;
    }
    return schema;

}

void
free_schema(Schema* schema)
{
    uint i = 0;

    if (schema == NULL) {
        return;
    }

    for (i = 0; i < schema->num_classes; ++i) {
        schema->num_examples_per_class[i] = 0;
    }
    schema->num_classes = 0;
    free(schema->num_examples_per_class); schema->num_examples_per_class = NULL;

    _clear_variable_info(schema->num_attributes, schema->attributes);
    schema->num_attributes = 0;
    free(schema->attributes);
    schema->attributes = NULL;
    free(schema);
}

void 
write_schema(FILE* fh, const Schema* schema)
{
    assert(fh != NULL);
    assert(schema != NULL);

    uint num_attr = schema->num_attributes;
    uint i;
    for (i = 0; i < num_attr; ++i) {
        VariableInfo* attr = &(schema->attributes[i]);
        fprintf(fh, "%s: ", attr->name);

        if (!attr->active) {
            fprintf(fh, "exclude\n");
            continue;
        }
        
        switch (attr->type) {
        case CONTINUOUS:
            fprintf(fh, "continuous");
            break;
        case DISCRETE:
            fprintf(fh, "discrete ");
            break;
        case CLASS:
            fprintf(fh, "class ");
            break;
        case EXCLUDE:
            fprintf(fh, "exclude");
            break;
        case UNKNOWN: // handle UNKNOWN same as default
        default:
            fprintf(fh, "ERROR, attribute %d has unexpected type", i);
            break;
        }

        if (attr->type == DISCRETE || attr->type == CLASS) {
            // Print out allowed values.
            assert(attr->arity > 0);
            fprintf(fh, " %s", attr->domain[0]);
            uint j;            
            for (j = 1; j < attr->arity; ++j) {
                fprintf(fh, ",%s", attr->domain[j]);
            }
        }

        fprintf(fh, "\n");
    }
}

uint
schema_num_classes(const Schema* schema)
{
    return schema->num_classes;
}

const char*
schema_class_name(const Schema* schema)
{
    return schema_attr_name(schema, schema->targetID);
}

const char* schema_get_class_value(const Schema* schema, uint valID)
{
    return schema_get_discrete_value(schema, schema->targetID, valID);
}

uint schema_class_column(const Schema* schema)
{
    return schema->targetID;
}

/*
uint 
schema_get_class_count(const Schema* schema, uint classID)
{
    assert(classID < schema->num_classes);
    return schema->num_examples_per_class[classID];
}

void 
schema_set_class_count(Schema* schema, uint classID, uint count)
{
    assert(classID < schema->num_classes);
    schema->num_examples_per_class[classID] = count;
}
*/


uint 
schema_num_attr(const Schema* schema)
{
    return schema->num_attributes;
}

const char* 
schema_attr_name(const Schema* schema, uint attrID)
{
    assert(attrID < schema->num_attributes);
    return schema->attributes[attrID].name;
}

Attribute_Type 
schema_attr_type(const Schema* schema, uint attrID)
{
    assert(attrID < schema->num_attributes);
    return schema->attributes[attrID].type;
}

// pre: attrID has type DISCRETE
uint 
schema_attr_arity(const Schema* schema, uint attrID)
{
    assert(attrID < schema->num_attributes);
    assert(DISCRETE == schema->attributes[attrID].type
        || CLASS == schema->attributes[attrID].type);
    return schema->attributes[attrID].arity;
}

const char* 
schema_get_discrete_value(const Schema* schema, uint attrID, uint valID)
{
    assert(attrID < schema->num_attributes);
    assert(DISCRETE == schema->attributes[attrID].type
        || CLASS == schema->attributes[attrID].type);
    assert(valID < schema->attributes[attrID].arity);
    return schema->attributes[attrID].domain[valID];
}

Missing_Value 
schema_get_attr_MV(const Schema* schema, uint attrID)
{
    assert(attrID < schema->num_attributes);
    return schema->attributes[attrID].imputed_value;
}

void 
schema_set_attr_MV(Schema* schema, uint attrID, Missing_Value mv)
{
    assert(attrID < schema->num_attributes);
    schema->attributes[attrID].imputed_value = mv;
}

Boolean 
schema_attr_is_active(const Schema* schema, uint attrID)
{
    assert(attrID < schema->num_attributes);
    return schema->attributes[attrID].active;
}


/************** PRIVATE MODULE HELPERS ****************/


void _clear_variable_info(uint num_vars, VariableInfo* vi)
{
    uint i,j;
    if (num_vars > 0) {
        assert(vi != NULL);
        for (i = 0; i < num_vars; ++i) {
            VariableInfo* varInfo = &vi[i];
            free(varInfo->name);
            varInfo->name = NULL;
            for (j = 0; j < varInfo->arity; ++j) {
                free(varInfo->domain[j]);
                varInfo->domain[j] = NULL;
            }
            free(varInfo->domain);
            varInfo->type = UNKNOWN;
            varInfo->arity = 0;
            varInfo->domain = NULL;
            varInfo->active = FALSE;
        }
    }
}

Boolean _parse_attribute(char* buffer, VariableInfo* result)
{
    assert(result != NULL);
    assert(buffer != NULL);

    Boolean success = TRUE;
    const char CONT_MARKER[] = "continuous";
    const char DISC_MARKER[] = "discrete";
    const char EXCL_MARKER[] = "exclude";
    const char CLS_MARKER[] = "class";

    // do while(false)
    do {
        // Find separator between attr name and its type information.
        char* end_name = strchr(buffer, ':');
        assert(end_name != NULL);

        // Save attribute name.
        size_t name_length = end_name - buffer;
        char* name = (char*)e_calloc(name_length+1, sizeof(char));
        strncpy(name, buffer, name_length);
        name[name_length] = '\0';
        strip_lt_whitespace(name);
        result->name = name;

        // Extract type information.
        char* type_info = end_name+1;
        strip_lt_whitespace(type_info);
        if (0 == strcmp(type_info, CONT_MARKER)) {  // exact comparison b/c of strip whitespace
            result->type = CONTINUOUS;
            result->arity = 0;
            result->domain = NULL;
            result->active = TRUE;
        }
        else if (type_info == strstr(type_info, EXCL_MARKER)) {
            // Verify that there is white space after the marker or an end of string.
            const char* end_marker = 
                type_info + sizeof(EXCL_MARKER)/sizeof(EXCL_MARKER[0]) -1;
            if (*end_marker != '\0' && !isspace(*end_marker)) {
                fprintf(stderr, "error: unexpected attribute specification format: '%s'\n", buffer);
                success = FALSE;
                break;
            }
            result->type = EXCLUDE;
            result->arity = 0;
            result->domain = NULL;
            result->active = FALSE;
        }
        else if (type_info == strstr(type_info, DISC_MARKER)) {
            // Verify that there is white space after the marker.
            const char* end_marker =
                type_info + sizeof(DISC_MARKER)/sizeof(DISC_MARKER[0]) -1;
            if (!isspace(*end_marker)) {
                fprintf(stderr, "error: unexpected attribute specification format: '%s'\n", buffer);
                success = FALSE;
                break;
            }
            result->type = DISCRETE;

            // Parse the allowed attribute values.
            const char* allowed_values = end_marker + 1;
            uint num_tokens = 0;
            char** tokens = _tokenize(allowed_values, ",", &num_tokens);
            if (tokens == NULL) {
                fprintf(stderr, "error: failed to parse specification '%s'\n", buffer);
                success = FALSE;
                break;
            }
            result->arity = num_tokens;
            result->domain = tokens;
            uint i;
            for (i = 0; i < num_tokens; ++i) {
                strip_lt_whitespace(tokens[i]);
                if (tokens[i][0] == '\0') {
                    // uh-oh, an empty token
                    fprintf(stderr, "error: white space values not allowed: '%s'\n", buffer);
                    success = FALSE;
                    break;
                }
            }
            result->active = TRUE;
        }
        else if (type_info == strstr(type_info, CLS_MARKER)) {
            // Verify that there is white space after the marker.
            const char* end_marker =
                type_info + sizeof(CLS_MARKER)/sizeof(CLS_MARKER[0]) -1;
            if (!isspace(*end_marker)) {
                fprintf(stderr, "error: unexpected class specification format: '%s'\n", buffer);
                success = FALSE;
                break;
            }
            result->type = CLASS;

            // Parse the allowed class values.
            const char* allowed_values = end_marker + 1;
            uint num_tokens = 0;
            char** tokens = _tokenize(allowed_values, ",", &num_tokens);
            if (tokens == NULL) {
                fprintf(stderr, "error: failed to parse specification '%s'\n", buffer);
                success = FALSE;
                break;
            }
            result->arity = num_tokens;
            result->domain = tokens;
            uint i;
            for (i = 0; i < num_tokens; ++i) {
                strip_lt_whitespace(tokens[i]);
                if (tokens[i][0] == '\0') {
                    // uh-oh, an empty token
                    fprintf(stderr, "error: white space values not allowed: '%s'\n", buffer);
                    success = FALSE;
                    break;
                }
            }
            result->active = TRUE;
        }
        else {
            // Unrecognized format.
            fprintf(stderr, "error: unrecognized attribute format: '%s'\n", buffer);
            success = FALSE;
            break;
        }

        // everything okay if we get here
    } while(FALSE);

    // If we failed to parse, release any memory we allocated.
    if (!success) {
        _clear_variable_info(1, result);
    }

    return success;
}

Boolean _set_target_info(Schema* schema, int manual_target)
{
    VariableInfo* attr = NULL;
    VariableInfo* done = NULL;

    // Manually set class variable, if requested.
    if (manual_target >= 0) {
        if (manual_target >= (int)(schema->num_attributes)) {
            fprintf(stderr, "error: target column %d > highest numbered column %d\n", 
                    manual_target+1, schema->num_attributes);
            return FALSE;
        }

        VariableInfo* target_attr = &(schema->attributes[manual_target]);

        // Mark any existing target variables as discrete instead.
        // This might be suprising if they are active, so print a warning.
        attr = schema->attributes;
        done = attr + schema->num_attributes;
        while (attr != done) {
            if (attr != target_attr && attr->type == CLASS) {
                attr->type = DISCRETE;
                if (attr->active) {
                    fprintf(
                        stderr, 
                        "warning: attr %s will be used as discrete input instead of target b/c of command line options\n",
                        attr->name);
                }
            }
            ++attr;
        }

        // Mark target variable as the CLASS variable.
        if (target_attr->type == DISCRETE || target_attr->type == CLASS) {
            target_attr->type = CLASS;
        }
        else {
            fprintf(stderr, "error: column %d is not a suitable type to be target\n", manual_target+1);
            return FALSE;
        }
    }

    // Sanity check: there should be exactly 1 *active* class variable.
    uint num_targets = 0;
    uint targetID = 0;
    uint i;
    for (i = 0; i < schema->num_attributes; ++i) {
        Boolean found_active_target = schema->attributes[i].active
            && schema->attributes[i].type == CLASS;
        if (found_active_target) {
            targetID = i;
            ++num_targets;
        }
    }
    if (num_targets == 0) {
        // Default: use the last active column.
        int target = -1;
        for (target = schema->num_attributes - 1; target >= 0; --target) {
            if (schema->attributes[target].active) {
                if (schema->attributes[target].type != DISCRETE) {
                    fprintf(stderr, "error: tried to use column %d as default truth column, but it is not discrete\n", target+1);
                    return FALSE;
                }
                schema->attributes[target].type = CLASS;
                targetID = target;
                num_targets = 1;
                break;
            }
        }
    }
    if (num_targets != 1) {
        fprintf(stderr, "error: found %d target variables (instead of 1).\n", num_targets);
        return FALSE;
    }

    // Initialize class specific meta data in schema.
    schema->targetID = targetID;
    schema->num_classes = schema->attributes[targetID].arity;
    schema->num_examples_per_class = e_calloc(schema->num_classes, sizeof(uint));

    return TRUE;
}

// REVIEW-2012-03-26-ArtM: Duplicates functionality in array.h?  Refactor or remove?
char** _tokenize(const char* s, const char* delim, uint* num_tokens)
{
    assert(s != NULL);
    assert(delim != NULL);
    assert(num_tokens != NULL);

    uint max_tokens = 10; // arbitrary initial guess for max number of tokens
    uint count = 0;
    char** tokens = (char**)e_calloc(max_tokens, sizeof(char*));

    // For each token ending with a delimiter...
    const char* start = s;
    const char* end = NULL;
    size_t length = 0;
    while (NULL != (end = strpbrk(start, delim))) {
        if (count >= max_tokens) {
            // Need more space to hold all the tokens.
            max_tokens *= 2;
            tokens = (char**)e_realloc(tokens, max_tokens*sizeof(char*));
        }

        // Make token from substring that spans [start,end).
        length = end - start;
        tokens[count] = (char*)e_calloc(length+1, sizeof(char));
        strncpy(tokens[count], start, length);
        tokens[count][length] = '\0';
        ++count;

        // Next token starts past the delimiter.
        start = end + 1;
    }

    // And the final token is the remainder of the string.
    if (count >= max_tokens) {
        max_tokens += 1;
        tokens = (char**)e_realloc(tokens, max_tokens*sizeof(char*));
    }
    tokens[count] = e_strdup(start);
    ++count;
    
    // Remove any unused token capacity.
    if (max_tokens > count) {
        tokens = (char**)e_realloc(tokens, count*sizeof(char*));
    }

    // Return results.
    *num_tokens = count;
    return tokens;
}
