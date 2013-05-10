# Zoltan
# This exports the variables CPPFLAGS_ZOLTAN and LIBS_ZOLTAN


AC_DEFUN([ACX_ZOLTAN], [

AC_SUBST(LIBS_ZOLTAN)
AC_SUBST(CPPFLAGS_ZOLTAN)

LIBZOLTAN="zoltan"

AC_ARG_WITH(zoltan, [AS_HELP_STRING([--with-zoltan=ZOLTAN_ROOT], [root installation of Zoltan])])
if test -n "$with_zoltan" ; then
  LIBS_ZOLTAN="-L$with_zoltan/lib"
  CPPFLAGS_ZOLTAN="-I$with_zoltan/include"
fi

AC_ARG_WITH(trilinos, [AS_HELP_STRING([--with-trilinos=TRILINOS_ROOT], [root installation of trilinos])])
if test -n "$with_trilinos" ; then
  LIBZOLTAN="trilinos_zoltan"
  if test -d "$with_trilinos" ; then
    LIBS_ZOLTAN="-L$with_trilinos/lib"
  fi
  CPPFLAGS_ZOLTAN="-I$with_trilinos/include/trilinos"
fi

acx_zoltan_save_CPPFLAGS="$CPPFLAGS"
acx_zoltan_save_LIBS="$LIBS"

CPPFLAGS="$CPPFLAGS $CPPFLAGS_ZOLTAN"
AC_CHECK_HEADERS(zoltan.h,AC_MSG_NOTICE([found zoltan.h]),[
  CPPFLAGS_ZOLTAN="-I/usr/include/trilinos"
  CPPFLAGS="$CPPFLAGS $CPPFLAGS_ZOLTAN"
  $as_unset AS_TR_SH([ac_cv_header_zoltan_h])
  AC_CHECK_HEADERS(zoltan.h,AC_MSG_NOTICE([found zoltan.h]),AC_MSG_ERROR([cannot find zoltan.h]))
])

LIBS="$acx_zoltan_save_LIBS $LIBS_ZOLTAN -l$LIBZOLTAN"
AC_CHECK_FUNCS(Zoltan_Initialize,AC_DEFINE(HAVE_ZOLTAN,1),[check_trilinos=yes])

if test x"$check_trilinos" == x"yes"; then
  LIBZOLTAN="trilinos_zoltan"
  LIBS="$acx_zoltan_save_LIBS $LIBS_ZOLTAN -l$LIBZOLTAN"

  $as_unset AS_TR_SH([ac_cv_func_Zoltan_Initialize])
  AC_CHECK_FUNCS(Zoltan_Initialize,AC_DEFINE(HAVE_ZOLTAN,1),AC_MSG_ERROR([cannot find zoltan]))
fi
LIBS_ZOLTAN="$LIBS_ZOLTAN -l$LIBZOLTAN"

CPPFLAGS="$acx_zoltan_save_CPPFLAGS"
LIBS="$acx_zoltan_save_LIBS"

]) dnl ACX_ZOLTAN
