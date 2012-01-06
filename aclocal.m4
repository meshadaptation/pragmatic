AC_DEFUN([ACX_zoltan], [
# Set variables...
AC_ARG_WITH(
	[zoltan],
        [AC_HELP_STRING([--with-zoltan],
                	[Prefix where Zoltan is installed.])],
	[zoltan="$withval"], [])

tmpLIBS=$LIBS
tmpCPPFLAGS=$CPPFLAGS
if test "x$zoltan" != "xno"; then
if test "x$zoltan" != "xyes"; then
zoltan_LIBS_PATH="$zoltan/lib"
zoltan_INCLUDES_PATH="$zoltan/include"
# Ensure the comiler finds the library...
tmpLIBS="$tmpLIBS -L$zoltan_LIBS_PATH"
tmpCPPFLAGS="$tmpCPPFLAGS  -I/$zoltan_INCLUDES_PATH"
fi
tmpLIBS="$tmpLIBS -L/usr/lib -L/usr/local/lib/ -lzoltan -lparmetis -lmetis $ZOLTAN_DEPS"
tmpCPPFLAGS="$tmpCPPFLAGS -I/usr/include/ -I/usr/local/include/"
fi
LIBS=$tmpLIBS
CPPFLAGS=$tmpCPPFLAGS
# Check that the compiler uses the library we specified...
if test -e $zoltan_LIBS_PATH/libzoltan.a; then
	echo "note: using $zoltan_LIBS_PATH/libzoltan.a"
fi 

# Check that the compiler uses the include path we specified...
if test -e $zoltan_INCLUDES_PATH/zoltan.mod; then
	echo "note: using $zoltan_INCLUDES_PATH/zoltan.mod"
fi 


AC_LANG_SAVE
AC_LANG_C
AC_CHECK_LIB(
	[zoltan],
	[Zoltan_Initialize],
	[AC_DEFINE(HAVE_ZOLTAN,1,[Define if you have zoltan library.])],
	[AC_MSG_ERROR( [Could not link in the zoltan library... exiting] )] )
# Save variables...
AC_LANG_RESTORE

ZOLTAN="yes"
AC_SUBST(ZOLTAN)

echo $LIBS
])dnl ACX_zoltan

# ===========================================================================
#            http://autoconf-archive.cryp.to/ax_check_define.html
# ===========================================================================
#
# SYNOPSIS
#
#   AC_CHECK_DEFINE([symbol], [ACTION-IF-FOUND], [ACTION-IF-NOT])
#   AX_CHECK_DEFINE([includes],[symbol], [ACTION-IF-FOUND], [ACTION-IF-NOT])
#
# DESCRIPTION
#
#   Complements AC_CHECK_FUNC but it does not check for a function but for a
#   define to exist. Consider a usage like:
#
#    AC_CHECK_DEFINE(__STRICT_ANSI__, CFLAGS="$CFLAGS -D_XOPEN_SOURCE=500")
#
# LICENSE
#
#   Copyright (c) 2008 Guido U. Draheim <guidod@gmx.de>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AC_DEFUN([AC_CHECK_DEFINED],[
AS_VAR_PUSHDEF([ac_var],[ac_cv_defined_$1])dnl
AC_CACHE_CHECK([for $1 defined], ac_var,
AC_TRY_COMPILE(,[
  #ifdef $1
  int ok;
  #else
  choke me
  #endif
],AS_VAR_SET(ac_var, yes),AS_VAR_SET(ac_var, no)))
AS_IF([test AS_VAR_GET(ac_var) != "no"], [$2], [$3])dnl
AS_VAR_POPDEF([ac_var])dnl
])

AC_DEFUN([AX_CHECK_DEFINED],[
AS_VAR_PUSHDEF([ac_var],[ac_cv_defined_$2])dnl
AC_CACHE_CHECK([for $1 defined], ac_var,
AC_TRY_COMPILE($1,[
  #ifndef $2
  int ok;
  #else
  choke me
  #endif
],AS_VAR_SET(ac_var, yes),AS_VAR_SET(ac_var, no)))
AS_IF([test AS_VAR_GET(ac_var) != "no"], [$3], [$4])dnl
AS_VAR_POPDEF([ac_var])dnl
])

AC_DEFUN([AX_CHECK_FUNC],
[AS_VAR_PUSHDEF([ac_var], [ac_cv_func_$2])dnl
AC_CACHE_CHECK([for $2], ac_var,
dnl AC_LANG_FUNC_LINK_TRY
[AC_LINK_IFELSE([AC_LANG_PROGRAM([$1
                #undef $2
                char $2 ();],[
                char (*f) () = $2;
                return f != $2; ])],
                [AS_VAR_SET(ac_var, yes)],
                [AS_VAR_SET(ac_var, no)])])
AS_IF([test AS_VAR_GET(ac_var) = yes], [$3], [$4])dnl
AS_VAR_POPDEF([ac_var])dnl
])# AC_CHECK_FUNC


# ===========================================================================
#          http://www.gnu.org/software/autoconf-archive/ax_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_MPI([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro tries to find out how to compile programs that use MPI
#   (Message Passing Interface), a standard API for parallel process
#   communication (see http://www-unix.mcs.anl.gov/mpi/)
#
#   On success, it sets the MPICC, MPICXX, MPIF77, or MPIFC output variable
#   to the name of the MPI compiler, depending upon the current language.
#   (This may just be $CC/$CXX/$F77/$FC, but is more often something like
#   mpicc/mpiCC/mpif77/mpif90.) It also sets MPILIBS to any libraries that
#   are needed for linking MPI (e.g. -lmpi or -lfmpi, if a special
#   MPICC/MPICXX/MPIF77/MPIFC was not found).
#
#   Note that this macro should be used only if you just have a few source
#   files that need to be compiled using MPI. In particular, you should
#   neither overwrite CC/CXX/F77/FC with the values of
#   MPICC/MPICXX/MPIF77/MPIFC, nor assume that you can use the same flags
#   etc. as the standard compilers. If you want to compile a whole program
#   using the MPI compiler commands, use one of the macros
#   AX_PROG_{CC,CXX,FC}_MPI.
#
#   ACTION-IF-FOUND is a list of shell commands to run if an MPI library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run if it is not
#   found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_MPI.
#
# LICENSE
#
#   Copyright (c) 2008 Steven G. Johnson <stevenj@alum.mit.edu>
#   Copyright (c) 2008 Julian C. Cummings <cummings@cacr.caltech.edu>
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation, either version 3 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

#serial 8

AU_ALIAS([ACX_MPI], [AX_MPI])
AC_DEFUN([AX_MPI], [
AC_PREREQ(2.50) dnl for AC_LANG_CASE

AC_LANG_CASE([C], [
	AC_REQUIRE([AC_PROG_CC])
	AC_ARG_VAR(MPICC,[MPI C compiler command])
	AC_CHECK_PROGS(MPICC, mpicc hcc mpxlc_r mpxlc mpcc cmpicc, $CC)
	ax_mpi_save_CC="$CC"
	CC="$MPICC"
	AC_SUBST(MPICC)
],
[C++], [
	AC_REQUIRE([AC_PROG_CXX])
	AC_ARG_VAR(MPICXX,[MPI C++ compiler command])
	AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC hcp mpxlC_r mpxlC mpCC cmpic++, $CXX)
	ax_mpi_save_CXX="$CXX"
	CXX="$MPICXX"
	AC_SUBST(MPICXX)
],
[Fortran 77], [
	AC_REQUIRE([AC_PROG_F77])
	AC_ARG_VAR(MPIF77,[MPI Fortran 77 compiler command])
	AC_CHECK_PROGS(MPIF77, mpif77 hf77 mpxlf_r mpxlf mpf77 cmpifc, $F77)
	ax_mpi_save_F77="$F77"
	F77="$MPIF77"
	AC_SUBST(MPIF77)
],
[Fortran], [
	AC_REQUIRE([AC_PROG_FC])
	AC_ARG_VAR(MPIFC,[MPI Fortran compiler command])
	AC_CHECK_PROGS(MPIFC, mpif90 mpxlf95_r mpxlf90_r mpxlf95 mpxlf90 mpf90 cmpif90c, $FC)
	ax_mpi_save_FC="$FC"
	FC="$MPIFC"
	AC_SUBST(MPIFC)
])

if test x = x"$MPILIBS"; then
	AC_LANG_CASE([C], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
		[C++], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
		[Fortran 77], [AC_MSG_CHECKING([for MPI_Init])
			AC_LINK_IFELSE([AC_LANG_PROGRAM([],[      call MPI_Init])],[MPILIBS=" "
				AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)])],
		[Fortran], [AC_MSG_CHECKING([for MPI_Init])
			AC_LINK_IFELSE([AC_LANG_PROGRAM([],[      call MPI_Init])],[MPILIBS=" "
				AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)])])
fi
AC_LANG_CASE([Fortran 77], [
	if test x = x"$MPILIBS"; then
		AC_CHECK_LIB(fmpi, MPI_Init, [MPILIBS="-lfmpi"])
	fi
	if test x = x"$MPILIBS"; then
		AC_CHECK_LIB(fmpich, MPI_Init, [MPILIBS="-lfmpich"])
	fi
],
[Fortran], [
	if test x = x"$MPILIBS"; then
		AC_CHECK_LIB(fmpi, MPI_Init, [MPILIBS="-lfmpi"])
	fi
	if test x = x"$MPILIBS"; then
		AC_CHECK_LIB(mpichf90, MPI_Init, [MPILIBS="-lmpichf90"])
	fi
])
if test x = x"$MPILIBS"; then
	AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
fi
if test x = x"$MPILIBS"; then
	AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
fi

dnl We have to use AC_TRY_COMPILE and not AC_CHECK_HEADER because the
dnl latter uses $CPP, not $CC (which may be mpicc).
AC_LANG_CASE([C], [if test x != x"$MPILIBS"; then
	AC_MSG_CHECKING([for mpi.h])
	AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [MPILIBS=""
		AC_MSG_RESULT(no)])
fi],
[C++], [if test x != x"$MPILIBS"; then
	AC_MSG_CHECKING([for mpi.h])
	AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [MPILIBS=""
		AC_MSG_RESULT(no)])
fi],
[Fortran 77], [if test x != x"$MPILIBS"; then
	AC_MSG_CHECKING([for mpif.h])
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[      include 'mpif.h'])],[AC_MSG_RESULT(yes)], [MPILIBS=""
		AC_MSG_RESULT(no)])
fi],
[Fortran], [if test x != x"$MPILIBS"; then
	AC_MSG_CHECKING([for mpif.h])
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[      include 'mpif.h'])],[AC_MSG_RESULT(yes)], [MPILIBS=""
		AC_MSG_RESULT(no)])
fi])

AC_LANG_CASE([C], [CC="$ax_mpi_save_CC"],
	[C++], [CXX="$ax_mpi_save_CXX"],
	[Fortran 77], [F77="$ax_mpi_save_F77"],
	[Fortran], [FC="$ax_mpi_save_FC"])

AC_SUBST(MPILIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x = x"$MPILIBS"; then
        $2
        :
else
        ifelse([$1],,[AC_DEFINE(HAVE_MPI,1,[Define if you have the MPI library.])],[$1])
        :
fi
])dnl AX_MPI

## https://github.com/tgamblin/libra/blob/master/m4/lx_find_mpi.m4
#################################################################################################
# Copyright (c) 2010, Lawrence Livermore National Security, LLC.  
# Produced at the Lawrence Livermore National Laboratory  
# Written by Todd Gamblin, tgamblin@llnl.gov.
# LLNL-CODE-417602
# All rights reserved.  
# 
# This file is part of Libra. For details, see http://github.com/tgamblin/libra.
# Please also read the LICENSE file for further information.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the disclaimer below.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the disclaimer (as noted below) in the documentation and/or other materials
#    provided with the distribution.
#  * Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse
#    or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################################

#
# LX_FIND_MPI()
#  ------------------------------------------------------------------------
# This macro finds an MPI compiler and extracts includes and libraries from 
# it for use in automake projects.  The script exports the following variables:
#
# AC_DEFINE variables:
#     HAVE_MPI         AC_DEFINE'd to 1 if we found MPI
#
# AC_SUBST variables:
#     MPICC            Name of MPI compiler
#     MPI_CFLAGS       Includes and defines for MPI C compilation
#     MPI_CLDFLAGS     Libraries and library paths for linking MPI C programs
#
#     MPICXX           Name of MPI C++ compiler
#     MPI_CXXFLAGS     Includes and defines for MPI C++ compilation
#     MPI_CXXLDFLAGS   Libraries and library paths for linking MPI C++ programs
#
#     MPIF77           Name of MPI Fortran 77 compiler
#     MPI_F77FLAGS     Includes and defines for MPI Fortran 77 compilation
#     MPI_F77LDFLAGS   Libraries and library paths for linking MPI Fortran 77 programs
# 
#     MPIFC            Name of MPI Fortran compiler
#     MPI_FFLAGS       Includes and defines for MPI Fortran compilation
#     MPI_FLDFLAGS     Libraries and library paths for linking MPI Fortran programs
# 
# Shell variables output by this macro:
#     have_C_mpi       'yes' if we found MPI for C, 'no' otherwise
#     have_CXX_mpi     'yes' if we found MPI for C++, 'no' otherwise
#     have_F77_mpi     'yes' if we found MPI for F77, 'no' otherwise
#     have_F_mpi       'yes' if we found MPI for Fortran, 'no' otherwise
#
AC_DEFUN([LX_FIND_MPI], 
[
     AC_LANG_CASE(
     [C], [
         AC_REQUIRE([AC_PROG_CC])
         if [[ ! -z "$MPICC" ]]; then
             LX_QUERY_MPI_COMPILER(MPICC, [$MPICC], C)
         else
             LX_QUERY_MPI_COMPILER(MPICC, [mpicc mpiicc mpixlc mpipgcc], C)
         fi
     ],
     [C++], [    
         AC_REQUIRE([AC_PROG_CXX])
         if [[ ! -z "$MPICXX" ]]; then
             LX_QUERY_MPI_COMPILER(MPICXX, [$MPICXX], CXX)
         else
             LX_QUERY_MPI_COMPILER(MPICXX, [mpicxx mpiCC mpic++ mpig++ mpiicpc mpipgCC mpixlC], CXX)
         fi
     ],
     [F77], [
         AC_REQUIRE([AC_PROG_F77])
         if [[ ! -z "$MPIF77" ]]; then
             LX_QUERY_MPI_COMPILER(MPIF77, [$MPIF77], F77)
         else
             LX_QUERY_MPI_COMPILER(MPIF77, [mpif77 mpiifort mpixlf77 mpixlf77_r], F77)
         fi
     ],
     [Fortran], [
         AC_REQUIRE([AC_PROG_FC])
         if [[ ! -z "$MPIFC" ]]; then
             LX_QUERY_MPI_COMPILER(MPIFC, [$MPIFC], F)
         else
             mpi_default_fc="mpif95 mpif90 mpigfortran mpif2003"
             mpi_intel_fc="mpiifort"
             mpi_xl_fc="mpixlf95 mpixlf95_r mpixlf90 mpixlf90_r mpixlf2003 mpixlf2003_r"
             mpi_pg_fc="mpipgf95 mpipgf90"
             LX_QUERY_MPI_COMPILER(MPIFC, [$mpi_default_fc $mpi_intel_fc $mpi_xl_fc $mpi_pg_fc], F)
         fi
     ])
])


#
# LX_QUERY_MPI_COMPILER([compiler-var-name], [compiler-names], [output-var-prefix])
#  ------------------------------------------------------------------------
# AC_SUBST variables:
#     MPI_<prefix>FLAGS       Includes and defines for MPI compilation
#     MPI_<prefix>LDFLAGS     Libraries and library paths for linking MPI C programs
# 
# Shell variables output by this macro:
#     found_mpi_flags         'yes' if we were able to get flags, 'no' otherwise
#
AC_DEFUN([LX_QUERY_MPI_COMPILER],
[
     # Try to find a working MPI compiler from the supplied names
     AC_PATH_PROGS($1, [$2], [not-found])
     
     # Figure out what the compiler responds to to get it to show us the compile
     # and link lines.  After this part of the macro, we'll have a valid 
     # lx_mpi_command_line
     echo -n "Checking whether $$1 responds to '-showme:compile'... "
     lx_mpi_compile_line=`$$1 -showme:compile 2>/dev/null`
     if [[ "$?" -eq 0 ]]; then
         echo yes
         lx_mpi_link_line=`$$1 -showme:link 2>/dev/null`
     else
         echo no
         echo -n "Checking whether $$1 responds to '-showme'... "
         lx_mpi_command_line=`$$1 -showme 2>/dev/null`
         if [[ "$?" -ne 0 ]]; then
             echo no
             echo -n "Checking whether $$1 responds to '-compile-info'... "
             lx_mpi_compile_line=`$$1 -compile-info 2>/dev/null`
             if [[ "$?" -eq 0 ]]; then
                 echo yes
                 lx_mpi_link_line=`$$1 -link-info 2>/dev/null`
             else
                 echo no
                 echo -n "Checking whether $$1 responds to '-show'... "
                 lx_mpi_command_line=`$$1 -show 2>/dev/null`
                 if [[ "$?" -eq 0 ]]; then
                     echo yes
                 else
                     echo no
                 fi
             fi
         else
             echo yes
         fi
     fi
          
     if [[ ! -z "$lx_mpi_compile_line" -a ! -z "$lx_mpi_link_line" ]]; then
         lx_mpi_command_line="$lx_mpi_compile_line $lx_mpi_link_line"
     fi

     if [[ ! -z "$lx_mpi_command_line" ]]; then
         # Now extract the different parts of the MPI command line.  Do these separately in case we need to 
         # parse them all out in future versions of this macro.
         lx_mpi_defines=`    echo "$lx_mpi_command_line" | grep -o -- '\(^\| \)-D\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)'`
         lx_mpi_includes=`   echo "$lx_mpi_command_line" | grep -o -- '\(^\| \)-I\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)'`
         lx_mpi_link_paths=` echo "$lx_mpi_command_line" | grep -o -- '\(^\| \)-L\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)'`
         lx_mpi_libs=`       echo "$lx_mpi_command_line" | grep -o -- '\(^\| \)-l\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)'`
         lx_mpi_link_args=`  echo "$lx_mpi_command_line" | grep -o -- '\(^\| \)-Wl,\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)'`
         
         # Create variables and clean up newlines and multiple spaces
         MPI_$3FLAGS="$lx_mpi_defines $lx_mpi_includes"
         MPI_$3LDFLAGS="$lx_mpi_link_paths $lx_mpi_libs $lx_mpi_link_args"
         MPI_$3FLAGS=`  echo "$MPI_$3FLAGS"   | tr '\n' ' ' | sed 's/^[[ \t]]*//;s/[[ \t]]*$//' | sed 's/  +/ /g'`
         MPI_$3LDFLAGS=`echo "$MPI_$3LDFLAGS" | tr '\n' ' ' | sed 's/^[[ \t]]*//;s/[[ \t]]*$//' | sed 's/  +/ /g'`

         OLD_CPPFLAGS=$CPPFLAGS
         OLD_LIBS=$LIBS
         CPPFLAGS=$MPI_$3FLAGS
         LIBS=$MPI_$3LDFLAGS

         AC_TRY_LINK([#include <mpi.h>],
                     [int rank, size;
                      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                      MPI_Comm_size(MPI_COMM_WORLD, &size);],
                     [# Add a define for testing at compile time.
                      AC_DEFINE([HAVE_MPI], [1], [Define to 1 if you have MPI libs and headers.])
                      have_$3_mpi='yes'],
                     [# zero out mpi flags so we don't link against the faulty library.
                      MPI_$3FLAGS=""
                      MPI_$3LDFLAGS=""
                      have_$3_mpi='no'])

         # AC_SUBST everything.
         AC_SUBST($1)
         AC_SUBST(MPI_$3FLAGS)
         AC_SUBST(MPI_$3LDFLAGS)

         LIBS=$OLD_LIBS
         CPPFLAGS=$OLD_CPPFLAGS
     else
         Echo Unable to find suitable MPI Compiler. Try setting $1.
         have_$3_mpi='no'         
     fi
])

