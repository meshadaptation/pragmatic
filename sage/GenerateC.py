import string

def GenerateC(cython_math):
    c_math=cython_math
    ploc = -1
    while True:
        ploc = c_math.find('^', ploc+1)
        if ploc==-1:
            break
        is_exp = False
        lsquared = False
        rsquared = False

        # Find value
        lvalue = rvalue = ploc-1
        if c_math[rvalue]==")":
            bcnt=1
            for lvalue in range(rvalue-1, -1, -1):
                if c_math[lvalue]==")":
                    bcnt = bcnt+1
                elif c_math[lvalue]=="(":
                    bcnt = bcnt-1
                if bcnt==0:
                    break
        else:
            lsquared = True
            for lvalue in range(rvalue, -1, -1):
                if "+-*/( ".find(c_math[lvalue])!=-1:
                    lvalue = lvalue + 1
                    break
            if c_math[lvalue:rvalue+1]=="e":
                is_exp = True

        # Find power
        lpower = rpower = ploc+1
        if c_math[lpower]=="(":
            bcnt=1
            for rpower in range(lpower+1, len(c_math)):
                if c_math[rpower]=="(":
                    bcnt = bcnt+1
                elif c_math[rpower]==")":
                    bcnt = bcnt-1
                if bcnt==0:
                    break
        else:
            for rpower in range(lpower+1, len(c_math)):
                if "+-*/( ".find(c_math[rpower])!=-1:
                    rpower = rpower - 1
                    break
            if c_math[lpower:rpower+1]=="2":
                rsquared = True

        tmp_c_math = ""
        if lvalue>0:
            tmp_c_math = c_math[0:lvalue]
        if is_exp:
            tmp_c_math = tmp_c_math+"exp(%s)"%(c_math[lpower:rpower+1])
        elif rsquared and rsquared:
            tmp_c_math = tmp_c_math+"%s*%s"%(c_math[lvalue:rvalue+1], c_math[lvalue:rvalue+1])
        else:
            tmp_c_math = tmp_c_math+"pow(%s, %s)"%(c_math[lvalue:rvalue+1], c_math[lpower:rpower+1])
        if rpower<len(c_math)-1:
            tmp_c_math = tmp_c_math+c_math[rpower+1:]
        c_math = tmp_c_math

    # We could have lots of int/int here so lets add an explicit case.
    tmp_c_math = ""
    for i in range(len(c_math)):
        tmp_c_math = tmp_c_math+c_math[i]
        if c_math[i]=="/":
            tmp_c_math = tmp_c_math+"(double)"
    c_math = tmp_c_math

    return c_math
