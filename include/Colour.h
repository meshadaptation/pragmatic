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

#ifndef COLOUR_H
#define COLOUR_H

#include <map>
#include <set>
#include <vector>
#include "confdefs.h"
#ifdef HAVE_ZOLTAN
#include "zoltan_cpp.h"
#endif

/*! \brief Class contains various methods for colouring undirected graphs.
 */
template<typename index_t>
class Colour{
 public:
  /*! This routine colours a undirected graph using the greedy colouring algorithm.
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void greedy(std::vector< std::set<index_t> > &NNList, index_t *colour){
    size_t NNodes = NNList.size();

    colour[0] = 0;
    for(size_t node=1;node<NNodes;node++){
      std::set<index_t> used_colours;
      for(typename std::set<index_t>::const_iterator it=NNList[node].begin();it!=NNList[node].end();++it)
        if(*it<(int)node)
          used_colours.insert(colour[*it]);

      for(index_t i=0;;i++)
        if(used_colours.count(i)==0){
          colour[node] = i;
          break;
        }
    }
  }
 private:
};
#endif
