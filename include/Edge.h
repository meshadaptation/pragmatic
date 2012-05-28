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

#ifndef H_EDGE
#define H_EDGE

#include "pragmatic_config.h"

/*! \brief Mesh edge object.
 */
template<typename index_t> class Edge{
 public:
  /*! Constructor
   * @param nid0 Node-Id 0
   * @param nid1 Node-Id 1
   */
  Edge(index_t nid0, index_t nid1){
    edge.first = std::min(nid0, nid1);
    edge.second = std::max(nid0, nid1);
  }
  
  /*! Copy constructor
   * @param in Edge object.
   */
  Edge(const Edge& in){
    *this = in;
  }
  
  /// Destructor
  ~Edge(){}
  
  /// Assignment operator
  Edge& operator=(const Edge &in){
    edge = in.edge;

    return *this;
  }
  
  /// Equality operator.
  bool operator==(const Edge& in) const{
    return this->edge == in.edge;
  }

  /// Inequality operator.
  bool operator!=(const Edge& in) const{
    return this->edge != in.edge;
  }

  /// Less-than operator
  bool operator<(const Edge& in) const{
    return this->edge < in.edge;
  }

  index_t connected(const Edge& in) const{
    if((edge.first==in.edge.first)||(edge.first==in.edge.second))
      return edge.first;
    else if((edge.second==in.edge.first)||(edge.second==in.edge.second))
      return edge.second;
    return -1;
  }
  
  bool contains(index_t nid) const{
    return (nid==edge.first)||(nid==edge.second);
  }

  template<typename _real_t, typename _index_t> friend class Mesh;
  template<typename _real_t, typename _index_t> friend class Surface;
  template<typename _real_t, typename _index_t> friend class Coarsen;
  template<typename _real_t, typename _index_t> friend class Swapping;
  template<typename _real_t, typename _index_t> friend class Refine;

 private:

  std::pair<index_t, index_t> edge;
};

#endif

