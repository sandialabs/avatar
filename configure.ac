
dnl **
dnl ** Package string appears in generated configure script
dnl ** and when configure is invoked with --help
dnl **

AC_INIT([avatar],[1.1])

dnl **
dnl ** Insert GNU Public License
dnl **

dnl **
dnl ** Make sure we're using at least this version of autoconf.
dnl **

AC_PREREQ(2.57)

dnl **
dnl ** Specify the language so that autoconf uses the appropriate preproccesr
dnl **

AC_LANG(C)
AC_PROG_CC
AC_C_CONST
AC_C_INLINE

dnl **
dnl ** Tell configure where the utilities dir is. This will be 
dnl ** used by this and all spawned configure processes.
dnl **

AC_CONFIG_AUX_DIR(scripts)

dnl **
dnl ** Make sure we are using GNU make.
dnl **

#CONFIGURE_GNU_MAKE

dnl **
dnl ** Determine the compiler
dnl **

#CONFIGURE_COMPILER

dnl **
dnl ** determine the available system install program.
dnl **

AC_PROG_INSTALL
CONFIGURE_CHECK_LIBS
CONFIGURE_CHECK_SIZE

dnl **
dnl ** find exodus
dnl **

#CONFIGURE_EXODUS

dnl **
dnl ** find netcdf
dnl **

#CONFIGURE_NETCDF

dnl **
dnl ** find fclib
dnl **

CONFIGURE_FCLIB

dnl **
dnl ** Create the output files
dnl **

#CONFIGURE_LIBXML2
CONFIGURE_GNU_GETOPT
CONFIGURE_GCOV
CONFIGURE_GPROF
CONFIGURE_DEBUG
CONFIGURE_WFLAGS

CONFIGURE_MPI

AC_CONFIG_FILES(Makefile src/gnu/Makefile src/Makefile tools/Makefile man/man1/Makefile util/Makefile unittest/Makefile regtest/Makefile)

AC_OUTPUT
