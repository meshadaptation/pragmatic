
#include <cmath>
#include <cfloat>
#include <iostream>

#include <generate_Steiner_ellipse_3d.h>

int main()
{
    double x1[]= { 1,  0, -4/sqrt(2)};
    double x2[]= {-1,  0, -4/sqrt(2)};
    double x3[]= { 0,  2,  4/sqrt(2)};
    double x4[]= { 0, -2,  4/sqrt(2)};
    double sm[6];
    pragmatic::generate_Steiner_ellipse(x1, x2, x3, x4, sm);

    // Test
    if(fabs(sm[0]-1./4)<10*DBL_EPSILON && fabs(sm[1])<10*DBL_EPSILON       && fabs(sm[2])<10*DBL_EPSILON &&
            fabs(sm[3]-1./16)<10*DBL_EPSILON && fabs(sm[4])<DBL_EPSILON &&
            fabs(sm[5]-1./64)<10*DBL_EPSILON) {
        std::cout<<"pass"<<std::endl;
    } else {
        std::cout<<"fail"<<std::endl;
    }

    return 0;
}
