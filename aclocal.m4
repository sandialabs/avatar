AC_DEFUN(CONFIGURE_CHECK_SIZE,
  [
    AC_CHECK_SIZEOF(int)
    AC_CHECK_SIZEOF(float)
    AC_CHECK_SIZEOF(double)
  ])

AC_DEFUN(CONFIGURE_FCLIB,
  [
  AC_ARG_WITH([fclib],
	[AC_HELP_STRING([--with-fclib=PATH],[Specify path to fclib installation])],
	[
	if test "x$with_fclib" = "xyes" ; then
                AC_MSG_CHECKING([fclib])
		fclib_version="`fcinfo --version`"
		if test "x$fclib_version" = "x"; then
                        AC_MSG_WARN([If you just installed fclib you may need to do a rehash to put fcinfo in your path])
			AC_MSG_ERROR([No fclib installation found. Use --with-fclib=<path to fclib top level dir>])
		else
			CFLAGS="`fcinfo --cflags` $CFLAGS"
			LIBS="`fcinfo --libs` $LIBS"
			AC_MSG_RESULT([v$fclib_version])
		fi
	else
		AC_CHECK_FILE([$with_fclib/bin/fcinfo], [
                        AC_MSG_CHECKING([fclib])
                        fclib_version="`$with_fclib/bin/fcinfo --version`"
			CFLAGS="`$with_fclib/bin/fcinfo --cflags` $CFLAGS"
			LIBS="`$with_fclib/bin/fcinfo --libs` $LIBS"
                        AC_MSG_RESULT([v$fclib_version])
		], [
                     AC_MSG_WARN([If you just installed fclib you may need to do a rehash to put fcinfo in your path])
                     AC_MSG_ERROR([No fclib installation found in $with_fclib])
                   ])
	fi
  	],
        [
        AC_MSG_CHECKING([fclib])
		fclib_version="`fcinfo --version`"
		if test "x$fclib_version" = "x"; then
                        AC_MSG_WARN([If you just installed fclib you may need to do a rehash to put fcinfo in your path])
			AC_MSG_ERROR([No fclib installation found. Use --with-fclib=<path to fclib top level dir>])
		else
			CFLAGS="`fcinfo --cflags` $CFLAGS"
			LIBS="`fcinfo --libs` $LIBS"
			AC_MSG_RESULT([v$fclib_version])
		fi
  ])
  
  dnl ** Now actually test the library and fc.h
  AC_CHECK_LIB([fc], [fc_initLibrary], [], [AC_MSG_ERROR([fclib library seems to be broken])])
  AC_MSG_CHECKING([for functional fc.h])
  AC_TRY_COMPILE([#include <fc.h>],[],[AC_MSG_RESULT(yes)],[AC_MSG_ERROR([fc.h seems to be broken])])
])

AC_DEFUN(CONFIGURE_GNU_GETOPT,
  [

  AC_CHECK_FUNCS([getopt_long], [
                                        HAVE_GETOPT_LONG=1
                                        AC_SUBST(HAVE_GETOPT_LONG)
                                ], [
                                        CPPFLAGS="-I./gnu $CPPFLAGS"
                                        HAVE_GETOPT_LONG=""
                                        AC_SUBST(HAVE_GETOPT_LONG)
                                ])

])

dnl ** ============================================================================
dnl ** CONFIGURE_GCOV
dnl ** Sets the CFLAGS values for using gcov and turns off optimazation
dnl ** ============================================================================

AC_DEFUN(CONFIGURE_GCOV,
  [
    AC_ARG_ENABLE([gcov],
                  [AC_HELP_STRING([--enable-gcov],[Enable compiling for gcov])],
                  [
                    if test "x$enable_gcov" = "xyes" ; then
                      AC_MSG_NOTICE([Enabling gcov support])
                      CFLAGS="-fprofile-arcs -ftest-coverage $CFLAGS"
                      dnl **
                      dnl ** Remove -Ox from CFLAGS
                      dnl ** Obviously, this could be written better
                      dnl **
                      NEW_CFLAGS=""
                      for flag in $CFLAGS
                      do
                          case $flag in
                              -O)
                                  AC_MSG_WARN([Removing $flag from CFLAGS for gcov])
                                  ;;
                              -O1)
                                  AC_MSG_WARN([Removing $flag from CFLAGS for gcov])
                                  ;;
                              -O2)
                                  AC_MSG_WARN([Removing $flag from CFLAGS for gcov])
                                  ;;
                              -O3)
                                  AC_MSG_WARN([Removing $flag from CFLAGS for gcov])
                                  ;;
                              -Os)
                                  AC_MSG_WARN([Removing $flag from CFLAGS for gcov])
                                  ;;
                              -Oz)
                                  AC_MSG_WARN([Removing $flag from CFLAGS for gcov])
                                  ;;
                              *)
                                  NEW_CFLAGS="$flag $NEW_CFLAGS"
                                  ;;
                          esac
                      done
                      CFLAGS="$NEW_CFLAGS"
                      USE_GCOV="1"
                      AC_SUBST([USE_GCOV])
                    fi
                  ], [])
    dnl **
    dnl ** Define these either way so the 'clean' target gets them every time
    dnl **
    GCOV_FILES="*.gcda *.gcno *.gcov"
    AC_SUBST([GCOV_FILES])
  ])
                  
dnl ** ============================================================================
dnl ** CONFIGURE_GPROF
dnl ** Sets the CFLAGS values for using gprof and turns off optimazation
dnl ** ============================================================================

AC_DEFUN(CONFIGURE_GPROF,
  [
    AC_ARG_ENABLE([gprof],
                  [AC_HELP_STRING([--enable-gprof],[Enable compiling for gprof])],
                  [
                    if test "x$enable_gprof" = "xyes" ; then
                      AC_MSG_NOTICE([Enable gprof support])
                      CFLAGS="-pg $CFLAGS"
                    fi
                  ], [])
  ])

AC_DEFUN(CONFIGURE_DEBUG,
  [
    AC_ARG_ENABLE([debug],
                  [AC_HELP_STRING([--enable-debug],[Enable debugging])],
                  [
                    CFLAGS="-g $CFLAGS"
                  ],[])
  ])

AC_DEFUN(CONFIGURE_WFLAGS,
  [
  
        for flag in '-W' '-Wall' '-Wcast-align' '-Wcast-qual' '-Wformat' '-Winline' \
                    '-Wmissing-braces' '-Wmissing-declarations' '-Wmissing-prototypes' \
                    '-Wnested-externs' '-Wnewline-eof' '-Wno-long-long' '-Wno-trigraphs' \
                    '-Wparentheses' '-Wpointer-arith' '-Wredundant-decls' '-Wreturn-type' \
                    '-Wshadow' '-Wsign-compare' '-Wstrict-prototypes' '-Wswitch' \
                    '-Wundef' '-Wunused-function' '-Wunused-label' '-Wunused-parameter' \
                    '-Wunused-value' '-Wunused-variable'; do
                AC_MSG_CHECKING([whether compiler accepts $flag])
                OLD_CFLAGS="$CFLAGS"
                CFLAGS="$flag $CFLAGS"
                AC_TRY_COMPILE([#include <fc.h>],[],[AC_MSG_RESULT(yes)],[
                        AC_MSG_RESULT(no)
                        CFLAGS="$OLD_CFLAGS"
                ])
        done
        AC_REQUIRE([AC_CANONICAL_HOST])
        AC_MSG_CHECKING([for aix])
        case $host_os in
                aix*)
                        # AIX - have to ask for large address or only get 256M for heap!
                        #  Ask for 7 segemnts (8 is the max but we leave one for debuggers
                        #  and other tools)
                        LDFLAGS="-Wl,-bmaxdata:0x70000000 $LDFLAGS"
                        AC_MSG_RESULT([yes])
                        ;;
                *)
                        AC_MSG_RESULT([no])
                        ;;
        esac
])

dnl ** This should probably try to remove the broken libraries from the LIBS but not sure how

AC_DEFUN(CONFIGURE_CHECK_LIBS,
  [
  AC_CHECK_LIB([m], [cos], [], [AC_MSG_ERROR([math library seems to be broken])])
  AC_CHECK_LIB([z], [gzopen], [], [AC_MSG_ERROR([zlib library seems to be broken])])
  AC_CHECK_LIB([pthread], [pthread_self], [], [])
  AC_CHECK_LIB([gsl], [gsl_rng_alloc], [], [
                                             MUST_COMPILE_GSL="1"
                                             AC_SUBST([MUST_COMPILE_GSL])
                                           ])
])
  
AC_DEFUN(CONFIGURE_MPI,
  [
  HAVE_MPI=""
  AC_ARG_WITH([force-mpi],
        [AC_HELP_STRING([--with-force-mpi=PATH],[Specify a path to MPI installation and do not check that it works])],
        [
        if test "x$with_force_mpi" = "xyes" ; then
            MPI_LIBS="-lmpi"
            HAVE_MPI="1"
            MPICC="mpicc"
            AC_MSG_WARN([Using libmpi without checking that it works])
            AC_SUBST([MPICC])
            AC_SUBST([MPI_LIBS])
        else
            MPI_LIBS="-lmpi"
            MPI_LDFLAGS="-L$with_force_mpi/lib"
            HAVE_MPI="1"
            MPI_CPPFLAGS="-I$with_force_mpi/include -I$with_force_mpi/inc"
            AC_MSG_WARN([Using $with_force_mpi/lib/libmpi without checking that it works])
            dnl ** Make sure mpicc exists
            AC_CHECK_FILE([$with_force_mpi/bin/mpicc], [
                  MPICC="$with_force_mpi/bin/mpicc"
                ],[AC_MSG_ERROR([No $with_force_mpi/bin/mpicc found])])
                
            AC_SUBST([MPICC])
            AC_SUBST([MPI_LDFLAGS])
            AC_SUBST([MPI_CPPFLAGS])
            AC_SUBST([MPI_LIBS])
        fi
        ],[])
  AC_ARG_WITH([force-mpich],
        [AC_HELP_STRING([--with-force-mpich=PATH],[Specify a path to MPICH installation and do not check that it works])],
        [
        if test "x$with_force_mpich" = "xyes" ; then
            MPI_LIBS="-lmpich"
            HAVE_MPI="1"
            MPICC="mpicc"
            AC_MSG_WARN([Using libmpich without checking that it works])
            AC_SUBST([MPICC])
            AC_SUBST([MPI_LIBS])
        else
            MPI_LIBS="-lmpich"
            MPI_LDFLAGS="-L$with_force_mpich/lib"
            HAVE_MPI="1"
            MPI_CPPFLAGS="-I$with_force_mpich/include -I$with_force_mpich/inc"
            AC_MSG_WARN([Using $with_force_mpich/lib/libmpich without checking that it works])
            dnl ** Make sure mpicc exists
            AC_CHECK_FILE([$with_force_mpich/bin/mpicc], [
                  MPICC="$with_force_mpich/bin/mpicc"
                ],[AC_MSG_ERROR([No $with_force_mpich/bin/mpicc found])])
                
            AC_SUBST([MPICC])
            AC_SUBST([MPI_LDFLAGS])
            AC_SUBST([MPI_CPPFLAGS])
            AC_SUBST([MPI_LIBS])
        fi
        ],[])
  AC_ARG_WITH([mpi],
	[AC_HELP_STRING([--with-mpi=PATH],[Specify path to MPI installation])],
	[
        if test "x$with_mpi" = "xyes" ; then
          dnl **
          dnl ** Try the system mpi installation
          dnl **
          dnl ** Check lib
          AC_CHECK_LIB([mpi], [MPI_Init], [
            MPI_LIBS="-lmpi"
            HAVE_MPI="1"
          ], [
            AC_CHECK_LIB([mpich], [PMPI_Init], [
              MPI_LIBS="-lmpich"
              HAVE_MPI="1"
            ], [AC_MSG_ERROR([mpi library seems to be broken])])
          ])
          dnl ** Check include
          AC_MSG_CHECKING([for functional mpi.h])
          AC_TRY_COMPILE([#include <mpi.h>],[],[
            AC_MSG_RESULT(yes)
          ],[AC_MSG_ERROR([mpi.h seems to be broken])])
        else
          dnl **
          dnl ** Try the mpi installation in the path supplied
          dnl **
          dnl ** Find lib
          AC_CHECK_FILE([$with_mpi/lib], [
                  OLD_LDFLAGS="$LDFLAGS"
                  LDFLAGS="-L$with_mpi/lib $LDFLAGS"
                  AC_CHECK_LIB([mpi], [MPI_Init], [
                    MPI_LIBS="-lmpi"
                    LDFLAGS="$OLD_LDFLAGS"
                    MPI_LDFLAGS="-L$with_mpi/lib"
                    HAVE_MPI="1"
                  ], [
                    AC_CHECK_LIB([mpich], [PMPI_Init], [
                      MPI_LIBS="-lmpich"
                      LDFLAGS="$OLD_LDFLAGS"
                      MPI_LDFLAGS="-L$with_mpi/lib"
                      HAVE_MPI="1"
                    ], [AC_MSG_ERROR([mpi library seems to be broken])
                  ])])
          ], [AC_MSG_ERROR([No $with_mpi/lib dir found])])
          dnl ** Find inc
          AC_CHECK_FILE([$with_mpi/inc], [
                  OLD_CPPFLAGS="$CPPFLAGS"
                  CPPFLAGS="-I$with_mpi/inc $CPPFLAGS"
                  AC_MSG_CHECKING([for functional mpi.h])
                  AC_TRY_COMPILE([#include <mpi.h>],[],[
                    AC_MSG_RESULT(yes)
                    CPPFLAGS="$OLD_CPPFLAGS"
                    MPI_CPPFLAGS="-I$with_mpi/inc"
                  ],[AC_MSG_ERROR([mpi.h seems to be broken])])
          ],[
              dnl ** Or check include
              AC_CHECK_FILE([$with_mpi/include], [
                      OLD_CPPFLAGS="$CPPFLAGS"
                      CPPFLAGS="-I$with_mpi/include $CPPFLAGS"
                      AC_MSG_CHECKING([for functional mpi.h])
                      AC_TRY_COMPILE([#include <mpi.h>],[],[
                        AC_MSG_RESULT(yes)
                        CPPFLAGS="$OLD_CPPFLAGS"
                        MPI_CPPFLAGS="-I$with_mpi/include"
                      ],[AC_MSG_ERROR([mpi.h seems to be broken])])
  	      ], [AC_MSG_ERROR([No $with_mpi/inc or $with_mpi/include dir found])])
          ])
          dnl ** Make sure mpicc exists
          AC_CHECK_FILE([$with_mpi/bin/mpicc], [
                MPICC="$with_mpi/bin/mpicc"
              ],[AC_MSG_ERROR([No $with_mpi/bin/mpicc found])])
              
          AC_SUBST([MPICC])
          AC_SUBST([MPI_LDFLAGS])
          AC_SUBST([MPI_CPPFLAGS])
          AC_SUBST([MPI_LIBS])
        fi
        ],[])
  AC_SUBST([HAVE_MPI])
])

AC_DEFUN(_CONFIGURE_MPI,
  [
  AC_ARG_ENABLE([mpi],
        [AC_HELP_STRING([--enable-mpi],[Build the MPI executables])], [
                        ACX_MPI
                        USE_MPI="yes"
                        AC_SUBST([USE_MPI])
                ], [])
])

