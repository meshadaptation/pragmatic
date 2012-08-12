/*  Copyright (C) 2010 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Gerard Gorman
 *  Applied Modelling and Computation Group
 *  Department of Earth Science and Engineering
 *  Imperial College London
 *
 *  g.gorman@imperial.ac.uk
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials provided
 *  with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 *  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 *  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */

#include <cmath>
#include <iostream>
#include <vector>

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"
#include "ticker.h"

#include "tinyxml.h"

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  // Benchmark times.
  double time_coarsen=0, time_refine=0, time_swap=0, time_smooth=0, time_adapt=0;

  int rank = MPI::COMM_WORLD.Get_rank();

  // For now only support a single process.
#ifndef NDEBUG
  int nprocs = MPI::COMM_WORLD.Get_size();
  assert(nprocs==1);
#endif

  int NNodes, NElements;
  Mesh<double, int> *mesh;
  {
    char pFilename[] = "../data/doughnut.xml\0";

    TiXmlDocument doc(pFilename);
    bool loadOkay = doc.LoadFile();

    if (!loadOkay){
      std::cerr<<"Failed to load file "<<pFilename<<std::endl;
      exit(-1);
    }

    TiXmlNode* header = doc.FirstChild();
    while(header != NULL and header->Type() != TiXmlNode::TINYXML_DECLARATION){
      header = header->NextSibling();
    }

    TiXmlNode* dolfin = doc.FirstChildElement();
    if(dolfin == NULL){
      std::cerr<<"Failed to find root node when loading options file."<<std::endl;
      exit(-1);
    }

    TiXmlNode* mesh_xmlnode = dolfin->FirstChildElement();
    std::string celltype(mesh_xmlnode->ToElement()->Attribute("celltype"));
    assert(celltype=="triangle");

    int dim;
    std::istringstream(mesh_xmlnode->ToElement()->Attribute("dim"))>>dim;
    assert(dim==2);

    // Get a handle on the mesh vertices and cells.
    TiXmlNode* xmlnode;
    TiXmlNode *vertices=NULL, *cells=NULL;
    for(xmlnode=mesh_xmlnode->FirstChildElement();xmlnode;xmlnode=xmlnode->NextSiblingElement()){
      if(xmlnode->ValueStr()=="vertices")
        vertices = xmlnode;
      if(xmlnode->ValueStr()=="cells")
        cells = xmlnode;
    }

    // Read in x, y.

    std::istringstream(vertices->ToElement()->Attribute("size"))>>NNodes;
    std::vector<double> x(NNodes), y(NNodes);

    for(xmlnode=vertices->FirstChildElement();xmlnode;xmlnode=xmlnode->NextSiblingElement()){
      int index;
      std::istringstream(xmlnode->ToElement()->Attribute("index"))>>index;
      std::istringstream(xmlnode->ToElement()->Attribute("x"))>>x[index];
      std::istringstream(xmlnode->ToElement()->Attribute("y"))>>y[index];
    }

    // Read in cells

    std::istringstream(cells->ToElement()->Attribute("size"))>>NElements;
    std::vector<int> Triangles(NElements*3);

    for(xmlnode=cells->FirstChildElement();xmlnode;xmlnode=xmlnode->NextSiblingElement()){
      int index;
      std::istringstream(xmlnode->ToElement()->Attribute("index"))>>index;
      int *n=&(Triangles[index*3]);

      std::istringstream(xmlnode->ToElement()->Attribute("v0"))>>n[0];
      std::istringstream(xmlnode->ToElement()->Attribute("v1"))>>n[1];
      std::istringstream(xmlnode->ToElement()->Attribute("v2"))>>n[2];
    }

    mesh = new Mesh<double, int>(NNodes, NElements, &(Triangles[0]), &(x[0]), &(y[0]));
  }

  // Stuff this into a pragmatic mesh.
  Surface<double, int> surface(*mesh);
  surface.find_surface();

  // Need to get the metric from the xml file - but for now just add this dummy.
  MetricField2D<double, int> metric_field(*mesh, surface);

  // Begin dummy...
  double I[] = {1.0, 0.0, 1.0};
  for(int i=0;i<NNodes;i++){
    double x = mesh->get_coords(i)[0];
    double y = mesh->get_coords(i)[1];

    I[0] = 10000.0*(x*x)/((x*x) + (y*y)) + 50.000*(y*y)/((x*x) + (y*y));
    I[1] = 9950.00*x*y/((x*x) + (y*y));
    I[2] = 50.000*(x*x)/((x*x) + (y*y)) + 10000.0*(y*y)/((x*x) + (y*y));

    metric_field.set_metric(I, i);
  }
  metric_field.update_mesh();

  // End dummy

  // Initial stats.
  double qmean = mesh->get_qmean();
  double qrms = mesh->get_qrms();
  double qmin = mesh->get_qmin();

  if(rank==0) std::cout<<"Initial quality:\n"
                       <<"Quality mean:  "<<qmean<<std::endl
                       <<"Quality min:   "<<qmin<<std::endl
                       <<"Quality RMS:   "<<qrms<<std::endl;
  VTKTools<double, int>::export_vtu("../data/test_dolfin_2d-initial", mesh);

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = sqrt(2.0);
  double L_low = L_up/2;

  Coarsen<double, int> coarsen(*mesh, surface);  
  Smooth<double, int> smooth(*mesh, surface);
  Refine<double, int> refine(*mesh, surface);
  Swapping2D<double, int> swapping(*mesh, surface);

  time_adapt = get_wtime();

  double tic = get_wtime();
  coarsen.coarsen(L_low, L_up);
  time_coarsen += get_wtime()-tic;

  double L_max = mesh->maximal_edge_length();

  double alpha = sqrt(2.0)/2;  
  for(size_t i=0;i<10;i++){
    double L_ref = std::max(alpha*L_max, L_up);

    tic = get_wtime();
    refine.refine(L_ref);
    time_refine += get_wtime() - tic;

    tic = get_wtime();
    coarsen.coarsen(L_low, L_ref);
    time_coarsen += get_wtime() - tic;

    if(rank==0) std::cout<<"INFO: Verify quality after refine/coarsen; but before swapping.\n";
    mesh->verify();

    tic = get_wtime();
    swapping.swap(0.95);
    time_swap += get_wtime() - tic;

    if(rank==0) std::cout<<"INFO: Verify quality after swapping.\n";
    mesh->verify();

    L_max = mesh->maximal_edge_length();

    if((L_max-L_up)<0.01)
      break;
  }

  std::vector<int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  tic = get_wtime();
  smooth.smooth("optimisation Linf", 200);
  time_smooth += get_wtime()-tic;

  time_adapt = get_wtime()-time_adapt;

  if(rank==0) std::cout<<"After optimisation based smoothing:\n";
  mesh->verify();

  VTKTools<double, int>::export_vtu("../data/test_dolfin_2d", mesh);

  qmean = mesh->get_qmean();
  qrms = mesh->get_qrms();
  qmin = mesh->get_qmin();

  std::cout<<"BENCHMARK: time_coarsen time_refine time_swap time_smooth\n";
  std::cout<<"BENCHMARK: "<<time_coarsen<<" "<<time_refine<<" "<<time_swap<<" "<<time_smooth<<"\n";

  if(rank==0){
    if((qmean>0.8)&&(qmin>0.3))
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail"<<std::endl;
  }

  // Write out new dolfin file.
  {
    NNodes = mesh->get_number_nodes();
    NElements = mesh->get_number_elements();

    TiXmlDocument wdoc;
    TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );  
    wdoc.LinkEndChild(decl); 

    TiXmlElement * root = new TiXmlElement("dolfin");  
    root->SetAttribute("xmlns:dolfin", "http://fenicsproject.org");
    wdoc.LinkEndChild( root );  

    TiXmlElement *wmesh_xmlnode = new TiXmlElement("mesh");  
    wmesh_xmlnode->SetAttribute("celltype", "triangle");
    wmesh_xmlnode->SetAttribute("dim", "2");
    root->LinkEndChild(wmesh_xmlnode);  

    TiXmlElement *wvertices_xmlnode = new TiXmlElement("vertices");  
    wvertices_xmlnode->SetAttribute("size", NNodes);
    wmesh_xmlnode->LinkEndChild(wvertices_xmlnode);

    for(int i=0;i<NNodes;i++){
      TiXmlElement *wvert_xmlnode = new TiXmlElement("vertex");  
      wvert_xmlnode->SetAttribute("index", i);
      const double *x = mesh->get_coords(i);
      wvert_xmlnode->SetAttribute("x", x[0]);
      wvert_xmlnode->SetAttribute("y", x[1]);
      wvertices_xmlnode->LinkEndChild(wvert_xmlnode);
    }

    TiXmlElement *wcells_xmlnode = new TiXmlElement("cells");  
    wcells_xmlnode->SetAttribute("size", NElements);
    wmesh_xmlnode->LinkEndChild(wcells_xmlnode);

    for(int i=0;i<NElements;i++){
      TiXmlElement *wcell_xmlnode = new TiXmlElement("triangle");  
      wcell_xmlnode->SetAttribute("index", i);
      const int *n = mesh->get_element(i);
      wcell_xmlnode->SetAttribute("v0", n[0]);
      wcell_xmlnode->SetAttribute("v1", n[1]);
      wcell_xmlnode->SetAttribute("v2", n[2]);

      wcells_xmlnode->LinkEndChild(wcell_xmlnode);
    }

    wdoc.SaveFile("../data/test_dolfin_2d.xml");  
  }

  delete mesh;
  MPI::Finalize();
}

