

#ifndef GMF_TOOLS_H
#define GMF_TOOLS_H

#include <vector>
#include <string>
#include <cfloat>
#include <typeinfo>

#include "Mesh.h"
#include "MetricTensor.h"
#include "ElementProperty.h"

extern "C" {
#ifdef HAVE_METIS
#include "metis.h"
#endif
}

#ifdef HAVE_MPI
#include "mpi_tools.h"
#endif

#ifdef HAVE_LIBMESHB
extern "C" {
#include <libmeshb7.h>
}
#endif



/*! \Toolkit for importing and exporting Gamma Mesh Format files. 
 *  This for now only works in serial.
 */
template<typename real_t> class GMFTools
{
public:

    static Mesh<real_t>* import_gmf_mesh(const char * meshName)
    {
        
        int             dim;
        char            fileName[128];
        long long       meshIndex;
        int             gmfVersion;

        strcpy(fileName, meshName);
        strcat(fileName, ".meshb");
        if ( !(meshIndex = GmfOpenMesh(fileName, GmfRead, &gmfVersion, &dim)) ) {
            strcpy(fileName, meshName);
            strcat(fileName,".mesh");
            if ( !(meshIndex = GmfOpenMesh(fileName, GmfRead, &gmfVersion, &dim)) ) {
                fprintf(stderr,"####  ERROR Mesh file %s.mesh[b] not found ", meshName);
                exit(1);
            }    
        }

        if (dim == 2)
            return import_gmf_mesh2d(meshIndex);    
        else if (dim == 3)
            return import_gmf_mesh3d(meshIndex);
        else {
            GmfCloseMesh(meshIndex);
            exit(45);
        }

        return NULL;

    }



    static Mesh<real_t>* import_gmf_metric(const char * meshname)
    {

    }

    static Mesh<real_t>* export_gmf_mesh(const char * meshname)
    {

    }

    static Mesh<real_t>* export_gmf_metric(const char * meshname)
    {

    }


private:

    static Mesh<real_t>* import_gmf_mesh2d(long long meshIndex)
    {

        int                 tag;
        std::vector<real_t> x, y;
        std::vector<int>    ENList;
        index_t             NNodes, NElements, bufTri[3];
        real_t              buf[2];
        Mesh<real_t>        *mesh=NULL;


        NNodes    = GmfStatKwd(meshIndex, GmfVertices);
        NElements = GmfStatKwd(meshIndex, GmfTriangles);
        x.reserve(NNodes);
        y.reserve(NNodes);
        ENList.reserve(3*NElements);

        if (NNodes <= 0 ) {
            fprintf(stderr, "####  ERROR  Number of vertices: %d <= 0\n", NNodes);
            exit(1);
        }

        GmfGotoKwd(meshIndex, GmfVertices);
        for(index_t i=0; i<NNodes; i++) {
            GmfGetLin(meshIndex, GmfVertices, &buf[0], &buf[1], &tag);
            x.push_back(buf[0]);
            y.push_back(buf[1]);
        }

        GmfGotoKwd(meshIndex, GmfTriangles);
        for(index_t i=0; i<NElements; i++) {
            GmfGetLin(meshIndex, GmfTriangles, &bufTri[0], &bufTri[1], &bufTri[2], &tag);
            for(int j=0; j<3; j++)
                ENList.push_back(bufTri[j]-1);
        }
        
        mesh = new Mesh<real_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));

        return mesh;

    }



    static Mesh<real_t>* import_gmf_mesh3d(long long meshIndex)
    {

        int                 tag;
        std::vector<real_t> x, y, z;
        std::vector<int>    ENList;
        index_t             NNodes, NElements, bufTet[4];
        real_t              buf[3];
        Mesh<real_t>        *mesh=NULL;


        NNodes    = GmfStatKwd(meshIndex, GmfVertices);
        NElements = GmfStatKwd(meshIndex, GmfTriangles);
        x.reserve(NNodes);
        y.reserve(NNodes);
        z.reserve(NNodes);
        ENList.reserve(3*NElements);

        if (NNodes <= 0 ) {
            fprintf(stderr, "####  ERROR  Number of vertices: %d <= 0\n", NNodes);
            exit(1);
        }

        GmfGotoKwd(meshIndex, GmfVertices);
        for(index_t i=0; i<NNodes; i++) {
            GmfGetLin(meshIndex, GmfVertices, &buf[0], &buf[1], &buf[2], &tag);
            x.push_back(buf[0]);
            y.push_back(buf[1]);
            z.push_back(buf[2]);
        }

        GmfGotoKwd(meshIndex, GmfTetrahedra);
        for(index_t i=0; i<NElements; i++) {
            GmfGetLin(meshIndex, GmfTetrahedra, &bufTet[0], &bufTet[1], &bufTet[2], &bufTet[3], &tag);
            for(int j=0; j<4; j++)
                ENList.push_back(bufTet[j]-1);
        }
        
        mesh = new Mesh<real_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));

        return mesh;

    }



};

#endif