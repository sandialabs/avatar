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
/**
 * \file checkall.c
 * \brief Main for all checks.
 *
 * \description
 *
 *    You can run everything or just desired modules by passing in their names
 *    as command line arguments (e.g. 'checkall' runs everything while 
 *    'checkall util base' runs just util and base). If you ask for specific
 *    modules, their will be an output line for each test. If you run
 *    everything the output will only provide lines for tests that fail.
 *
 *    Specifying -v will turn on FCLib warning messages and -V will turn
 *    on FCLib log messages.
 *
 *    Check runs each set of unit tests in it's own process. To turn this
 *    off (to debug) set the environment varible CK_FORK to no.
 *
 * $Source: /home/Repositories/avatar/avatar/unittest/checkall.c,v $
 * $Revision: 1.2 $ 
 * $Date: 2006/02/03 03:49:58 $
 *
 * \modifications
 *    - 12/20/04 WSK, created.
 *    - 01/19/05 WSK, added list of modules names for usage statement.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "check.h"
#include "fc.h"
#include "checkall.h"

static Suite *dummy_suite(void)
{
    Suite *suite = suite_create("");
    return suite;
}
 
// Look for the string in a list of strings - returns index in list   
static int findStringInList(char* query, int num, char** list) {
    int i;
    for (i = 0; i < num; i++) {
    if (!strcmp(query, list[i]))
        return i;
    }
    return -1; // didn't find it
}

int main (int argc, char** argv)
{
    int nf, i;
    SRunner *srunner;
    enum fork_status fork;
    enum print_output print_mode = CK_VERBOSE;
    int doAll = 0;       // 1 means do all
    char** nameList;
    int listLen;

    // These are in order roughly by dependency (i.e. base stuff first)
    // DEVELOPER NOTE: To add a module:
    //   o increment numModule
    //   o add name to moduleNames
    //   o add suite to moduleSuites
    //   o add a declaration for the suite to checkall.h
    // (Note that two cases are supported by conditional compiling, the first
    //  is the default case, the second is for tests that depend on external
    // data)
#ifndef EXTERNAL_DATA
    int numModule = 2;
    char moduleNames[][1024] = {
                                 "arrayutils",
                                 "ccregions"
                               };
    Suite *moduleSuites[] = { 
                              arrayutils_suite(),
                              ccregions_suite()
                            };
//#else
    // these are tests that use external data
//    int numModule = 2;
//    char moduleNames[][1024] = {
//                                 "safcover2", 
//                                 "fileio2"
//                               };
//  Suite *moduleSuites[] = { safcover2_suite(),
//			    fileio2_suite() };
#endif
  
  // handle arguments
  if (argc > 1 && (!strncasecmp(argv[1], "-h", sizeof(char)*2) ||
                   !strncasecmp(argv[1], "--h", sizeof(char)*3)) ) {
    printf("usage:\n");
    printf("   checkall               - runs unit tests for all modules, with\n");
    printf("                            output only for failed tests\n");
    printf("   checkall module1 [...] - runs unit tests for only the requested modules,\n");
    printf("                            with output for each test\n");
    printf("\n");
    printf("options (must be first argument):\n");
    printf("   -h = this usage message\n");
    printf("   -v = turns on FCLib warning messages\n");
    printf("   -V = turns on FCLib log messages\n");
    printf("\n");
    printf("Note: For debugging, you can prevent check from forking processes\n");
    printf("by setting the environment variable 'CK_FORK' to 'no'.\n");
    printf("\n");
    printf("Possible modules:\n");
    for (i = 0; i < numModule; i++)
      printf("   %s\n", moduleNames[i]);
    exit(0);
  }
  else if (argc > 1 && !strncasecmp(argv[1], "-v", sizeof(char)*2)) {
    listLen = argc-2;
    nameList = &argv[2];
    if (argv[1][1] == 'v')
      fc_messages = FC_WARNING_MESSAGES;
    else if (argv[1][1] == 'V')
      fc_messages = FC_LOG_MESSAGES;
  }
  else {
    fc_messages = FC_QUIET;
    listLen = argc-1;
    nameList = &argv[1];
  } 
  if (listLen == 0) {
    doAll = 1;
    print_mode = CK_NORMAL;
  }

  // create srunner & add suites
  srunner = srunner_create(dummy_suite());
  if (doAll) {
    for (i = 0; i < numModule; i++)
      srunner_add_suite(srunner, moduleSuites[i]);
  }
  else {
    int seen_one = 0, indx;
    char *tempList[numModule];
    for (i = 0; i < numModule; i++)
      tempList[i] = moduleNames[i];
    for (i = 0; i < listLen; i++) {
      indx = findStringInList(nameList[i], numModule, tempList);
      if (indx > -1) {
	seen_one = 1;
	srunner_add_suite(srunner, moduleSuites[indx]);
      }
      else
	printf("WARNING: No module named '%s'--ignoring\n", nameList[i]);
    }
    if (!seen_one) {
      printf("ERROR: no valid modules provided, call with -h for list of modules\n");
      exit(-1);
    }
  }

  // check fork status
  fork = srunner_fork_status(srunner);
  if (fork == CK_FORK)
    isForking = 1;
  else {
    isForking = 0;
    printf("***WARNING*** Check is running without forking\n");
    printf("**CK_NOFORK: Cannot do tests that involve reinitializing state\n");
    fc_setLibraryVerbosity(fc_messages);
    fc_initLibrary();
  }

  // run the tests - Output type normal if doAll, verbose if not
  srunner_run_all (srunner, print_mode);
  nf = srunner_ntests_failed(srunner);

  // if not forking cleanup
  if (!isForking) 
    fc_finalLibrary();

  // cleanup
  srunner_free(srunner);
  return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
