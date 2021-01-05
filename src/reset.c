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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <libgen.h>
#include "version_info.h"
#include "crossval.h"
#include "reset.h"

void reset_CV_Class(CV_Class *cvc) {

    cvc->class_var_name    = NULL;
    cvc->num_classes       = 0;
    cvc->thresholds        = NULL;
    cvc->class_frequencies = NULL;
    cvc->class_names       = NULL;

    return;

}

void reset_CV_Dataset(CV_Dataset *cvd) {

    reset_CV_Metadata(&(cvd->meta));
    cvd->examples = NULL;

    return ;

}

void reset_Exo_Data(Exo_Data *ed) {

    ed->num_seq_meshes = 0;
    ed->seq_meshes     = NULL;
    ed->assoc_type     = 0;
    ed->variables      = NULL; 

    return ;

} 

void reset_CV_Example(CV_Example *cve) {

    cve->global_id_num        = 0;
    cve->random_gid           = 0;
    cve->fclib_seq_num        = 0;
    cve->fclib_id_num         = 0;
    cve->containing_class_num = 0;
    cve->predicted_class_num  = 0;
    cve->containing_fold_num  = 0;
    cve->bl_clump_num         = 0;
    cve->distinct_attribute_values = NULL;

    return ;

}

void reset_CV_Metadata(CV_Metadata *cvm) {

    reset_Exo_Data(&(cvm->exo_data));
    cvm->num_classes    = 0;
    cvm->num_attributes = 0;
    cvm->num_fclib_seq  = 0; 
    cvm->num_examples   = 0;
    cvm->attribute_names        = NULL; 
    cvm->num_discrete_values    = NULL;
    cvm->discrete_attribute_map = NULL; 
    cvm->class_names            = NULL; 
    cvm->num_examples_per_class = NULL; 
    cvm->attribute_types        = NULL;
    cvm->global_offset          = NULL;     
    cvm->Missing                = NULL;

    return ;

} 

void reset_CV_Subset(CV_Subset *cvs) {

    reset_CV_Metadata(&(cvs->meta));
    cvs->malloc_examples  = 0;
    cvs->discrete_used    = NULL;
    cvs->float_data       = NULL;
    cvs->high             = NULL;
    cvs->low              = NULL; 
    cvs->smote_float_data = NULL;
    cvs->smote_high       = NULL;
    cvs->smote_low        = NULL;
    cvs->weights          = NULL;
    cvs->examples         = NULL;

    return ;

}

void reset_DT_Ensemble(DT_Ensemble *dte) {

    dte->Trees          = NULL;
    dte->boosting_betas = NULL;
    dte->Books          = NULL;
    dte->num_trees             = 0;
    dte->num_classes           = 0;
    dte->num_attributes        = 0;
    dte->num_training_examples = 0;
    dte->num_training_examples_per_class = NULL;
    dte->attribute_types                 = NULL;
    dte->weights                         = NULL;
    dte->Missing                         = NULL;
    return ;
}

