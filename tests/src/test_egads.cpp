#include "Mesh.h"

#include "ticker.h"


int main(int argc, char **argv)
{

    int tri[3] = {0, 1, 2};
    double x[3] = {0, 0, 1}, y[3] = {0, 1, 0};
    Mesh<double> mesh = Mesh<double>(3, 1, tri, x, y);
    mesh.analyzeCAD("../data/cube-cylinder.step");

    return 0;
}