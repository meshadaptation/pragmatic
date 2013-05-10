#include "cpragmatic.h"

Mesh<double, int>* _cpragmatic_mesh = NULL;
Surface<double, int>* _cpragmatic_surface = NULL;
MetricField<double, int>* _cpragmatic_metric_field = NULL;

extern "C" {
  void cpragmatic_initialise_2d(int* NNodes, int* NElements, int* enlist, double* x, double* y){
    assert(!_cpragmatic_mesh);
    assert(!_cpragmatic_surface);
    assert(!_cpragmatic_metric_field);

    _cpragmatic_mesh = new Mesh<double, int>(*NNodes, *NElements, enlist, x, y);
    assert(_cpragmatic_mesh);
    
    return;
  }

  void cpragmatic_set_surface(int* nfacets, int* facets, int* boundary_ids, int* coplanar_ids){
    assert(_cpragmatic_mesh);
    assert(!_cpragmatic_surface);

    _cpragmatic_surface = new Surface<double, int>(*_cpragmatic_mesh);
    _cpragmatic_surface->set_surface(*nfacets, facets, boundary_ids, coplanar_ids);

    return;
  }

  void cpragmatic_set_metric(double* metric){
    assert(_cpragmatic_mesh);
    assert(_cpragmatic_surface);

    if(!_cpragmatic_metric_field){
      _cpragmatic_metric_field = new MetricField<double, int>(*_cpragmatic_mesh, *_cpragmatic_surface);
    }

    _cpragmatic_metric_field->set_metric(metric);
    _cpragmatic_metric_field->update_mesh();

    return;
  }
  
  void cpragmatic_metric_add_field(double* psi, double* error, int* pnorm){
    assert(_cpragmatic_mesh);
    assert(_cpragmatic_surface);

    if(!_cpragmatic_metric_field){
      _cpragmatic_metric_field = new MetricField<double, int>(*_cpragmatic_mesh, *_cpragmatic_surface);
    }
    
    _cpragmatic_metric_field->add_field(psi, *error, *pnorm);
    _cpragmatic_metric_field->update_mesh();

    int n = _cpragmatic_mesh->get_number_nodes();
    int dim = _cpragmatic_mesh->get_number_dimensions();

    assert(dim == 2);
    double* metric_arr = (double*)malloc(n * 4 * sizeof(double));
    assert(metric_arr);
    _cpragmatic_metric_field->get_metric(metric_arr);
    for(size_t i = 0;i < n;i++){
      for(size_t j = 0;j < 2;j ++){
        if(fabs(metric_arr[i * 4    ] * metric_arr[i * 4 + 3] - metric_arr[i * 4 + 1] * metric_arr[i * 4 + 2]) < 1.0e-12){
          metric_arr[i * 4    ] += 1.0e-6;  metric_arr[i * 4 + 3] += 1.0e-6;
        }else{
          break;
        }
      }
    }
    _cpragmatic_metric_field->set_metric(metric_arr);
    _cpragmatic_metric_field->update_mesh();
    free(metric_arr);

    return;
  }
  
  void cpragmatic_apply_metric_gradation(double* gradation){
    assert(_cpragmatic_metric_field);

    _cpragmatic_metric_field->apply_gradation(*gradation);
    _cpragmatic_metric_field->update_mesh();

    return;
  }

  void cpragmatic_apply_metric_bounds(double* min_len, double* max_len){
    assert(_cpragmatic_metric_field);

    _cpragmatic_metric_field->apply_max_edge_length(*max_len);
    _cpragmatic_metric_field->apply_min_edge_length(*min_len);
    _cpragmatic_metric_field->update_mesh();

    return;
  }

  void cpragmatic_adapt(int* smooth){
    assert(_cpragmatic_mesh);
    assert(_cpragmatic_surface);
    
    // The following is taken from the example in the PRAgMaTIc manual

    // Set upper and lower tolerances on edge length as measured in metric space.
    double L_up = sqrt(2.0);
    double L_low = L_up/2;

    // Initialise adaptive modules
    Coarsen<double, int> coarsen(*_cpragmatic_mesh, *_cpragmatic_surface);
    Smooth<double, int> smoother(*_cpragmatic_mesh, *_cpragmatic_surface);
    Refine<double, int> refine(*_cpragmatic_mesh, *_cpragmatic_surface);
    Swapping<double, int> swapping(*_cpragmatic_mesh, *_cpragmatic_surface);

    // Apply initial coarsening
    coarsen.coarsen(L_low, L_up);

    // Initialise the maximum edge length.
    double L_max = _cpragmatic_mesh->maximal_edge_length();

    double alpha = sqrt(2.0)/2;
    for(size_t i=0;i<10;i++){
      std::cout << "Adapt iteration " << (i + 1) << " of 10" << std::endl;
      
      // Used to throttle the coarsening and refinement.
      double L_ref = std::max(alpha*L_max, L_up);

      // Refine mesh.
      refine.refine(L_ref);

      // Coarsen mesh.
      coarsen.coarsen(L_low, L_ref);

      // Improve quality through swapping
      swapping.swap(0.95);

      // Update the maximum edge length and check convergence of algorithm,
      L_max = _cpragmatic_mesh->maximal_edge_length();
      if((L_max-L_up)<0.01)
        break;
    }

    // Defragment the memory usage.
    std::map<int, int> active_vertex_map;
    _cpragmatic_mesh->defragment(&active_vertex_map);
    _cpragmatic_surface->defragment(&active_vertex_map);

    if(*smooth){
      // Apply vertex smoothing.
      smoother.smooth("optimisation Linf", 4);
    }
    
    return;
  }

  void cpragmatic_query_output(int* NNodes, int* NElements){
    assert(_cpragmatic_mesh);

    *NNodes = _cpragmatic_mesh->get_number_nodes();
    *NElements = _cpragmatic_mesh->get_number_elements();

    return;
  }

  void cpragmatic_get_output_2d(int* enlist, double* x, double* y){
    assert(_cpragmatic_mesh);

    assert(_cpragmatic_mesh->get_number_dimensions() == 2);
    for(size_t i = 0;i < _cpragmatic_mesh->get_number_elements();i++){
      const int* cell = _cpragmatic_mesh->get_element(i);
      enlist[i * 3    ] = cell[0];
      enlist[i * 3 + 1] = cell[1];
      enlist[i * 3 + 2] = cell[2];
    }
    for(size_t i = 0;i < _cpragmatic_mesh->get_number_nodes();i++){
      const double* coord = _cpragmatic_mesh->get_coords(i);
      x[i] = coord[0];
      y[i] = coord[1];
    }

    return;
  }

  void cpragmatic_finalise(void){
    if(_cpragmatic_mesh){
      delete(_cpragmatic_mesh);
      _cpragmatic_mesh = NULL;
    }

    if(_cpragmatic_surface){
      delete(_cpragmatic_surface);
      _cpragmatic_surface = NULL;
    }

    if(_cpragmatic_metric_field){
      delete(_cpragmatic_metric_field);
      _cpragmatic_metric_field = NULL;
    }

    return;
  } 
}