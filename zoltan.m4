AC_DEFUN([ACX_zoltan], [
# Set variables...
AC_ARG_WITH(
	[zoltan],
	[  --with-zoltan=prefix        Prefix where zoltan is installed],
	[zoltan="$withval"],
    [])

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
tmpLIBS="$tmpLIBS -L/usr/lib -L/usr/local/lib/ -lzoltan -lparmetis $ZOLTAN_DEPS"
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

