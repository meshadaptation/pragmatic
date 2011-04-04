/* 
 *    Copyright (C) 2010 Imperial College London and others.
 *    
 *    Please see the AUTHORS file in the main source directory for a full list
 *    of copyright holders.
 *
 *    Gerard Gorman
 *    Applied Modelling and Computation Group
 *    Department of Earth Science and Engineering
 *    Imperial College London
 *
 *    amcgsoftware@imperial.ac.uk
 *    
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation,
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
 *    USA
 */
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

#include <iostream>
#include <vector>

#include "Surface.h"
#include "Mesh.h"
#include "vtk_tools.h"

using namespace std;

/* Tests
   1. Assert number of coplanar id's is 4.
 */

int main(int argc, char **argv){
  Mesh<double, int> *mesh=NULL;
  import_vtu("../data/box20x20.vtu", mesh);
  
  Surface<double, int> surface(*mesh);

  export_vtu("../data/test_surface_2d.vtu", &surface);
  
  std::set<int> unique_ids;
  for(int i=0;i<surface.get_number_facets();i++){
    unique_ids.insert(surface.get_coplanar_id(i));
  }

  if(unique_ids.size()==4)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";

  delete mesh;
  
  return 0;
}
