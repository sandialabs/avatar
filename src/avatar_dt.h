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
#ifndef AVATAR_DT_H
#define AVATAR_DT_H

//-------------------------------------------------------------------------------------------
// 2020/08/03: Argument list changed by Cosmin Safta
// the original longer list lead to a compiler error with clang as the number of arguments
// is larger than the standard C allows. The extra three arguments are not currently
// needed and are thus commented out both here as well as in avatar_dt.c and dt.c
//-------------------------------------------------------------------------------------------
//int avatar_dt (int argc, char **argv, char* names_file, char* tree_file, char* test_data_file);
int avatar_dt (int argc, char **argv);

#endif // AVATAR_DT_H
