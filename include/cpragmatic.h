#ifndef CPRAGMATIC_H
#define CPRAGMATIC_H

#include <cassert>

// The order of these matters (!)
#include "Surface.h"
#include "Coarsen.h"

#include "Mesh.h"
#include "MetricField.h"
#include "Refine.h"
#include "Swapping.h"
#include "Smooth.h"

extern Mesh<double, int>* _cpragmatic_mesh;
extern Surface<double, int>* _cpragmatic_surface;
extern MetricField<double, int>* _cpragmatic_metric_field;

extern "C" {
  void cpragmatic_initialise_2d(int* NNodes, int* NElements, int* enlist, double* x, double* y);
  void cpragmatic_set_boundary(int* nfacets, int* facets, int* boundary_ids);
  void cpragmatic_set_metric(double* metric);
  void cpragmatic_metric_add_field(double* psi, double* error, int* pnorm);
  void cpragmatic_apply_metric_bounds(double* min_len, double* max_len);
  void cpragmatic_adapt(int* smooth);
  void cpragmatic_query_output(int* NNodes, int* NElements);
  void cpragmatic_get_output_2d(int* enlist, double* x, double* y);
  void cpragmatic_finalise(void);
}
  
#endif
