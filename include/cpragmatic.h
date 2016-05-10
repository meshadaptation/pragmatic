#ifndef CPRAGMATIC_H
#define CPRAGMATIC_H

#if defined(__cplusplus)
extern "C" {
#endif
void pragmatic_2d_init(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y);
void pragmatic_3d_init(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z);
void pragmatic_set_boundary(const int *nfacets, const int *facets, const int *ids);
void pragmatic_set_metric(const double *metric);
void pragmatic_add_field(const double *psi, const double *error, int *pnorm);
void pragmatic_adapt(void);
void pragmatic_coarsen(void);
void pragmatic_get_info(int *NNodes, int *NElements);
void pragmatic_get_coords_2d(double *x, double *y);
void pragmatic_get_coords_3d(double *x, double *y, double *z);
void pragmatic_get_elements(int *elements);
void pragmatic_get_boundaryTags(int ** tags);
void pragmatic_finalize(void);
#if defined(__cplusplus)
}
#endif
#endif
