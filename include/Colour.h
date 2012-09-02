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

#ifndef COLOUR_H
#define COLOUR_H

#include <map>
#include <set>
#include <vector>
#include <deque>

#include "pragmatic_config.h"

/*! \brief Performs a simple first breath greedy graph colouring of a local undirected graph.
 */
template<typename index_t>
class Colour{
 public:
  /*! This routine colours a undirected graph using the greedy colouring algorithm.
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void greedy(std::vector< std::vector<index_t> > &NNList, int *colour){
    size_t NNodes = NNList.size();
    int max_colour=64;

    // Colour first active node.
    size_t node;
    for(node=0;node<NNodes;node++){
      if(NNList[node].size()>0){
        colour[node] = 0;
        break;
      }
    }

    // Colour remaining active nodes.
    for(;node<NNodes;node++){
      if(NNList[node].size()==0)
        continue;

      std::vector<bool> used_colours(max_colour, false);;
      for(typename std::vector<index_t>::const_iterator it=NNList[node].begin();it!=NNList[node].end();++it){
        if(*it<(int)node){
          if(colour[*it]>=(signed)max_colour){
            max_colour*=2;
            used_colours.resize(max_colour, false);
          }

          used_colours[colour[*it]] = true;
        }
      }

      for(index_t i=0;;i++)
        if(!used_colours[i]){
          colour[node] = i;
          break;
        }
    }
  }

  /*! This routine colours a undirected graph using the greedy colouring algorithm.
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param active indicates which nodes are turned on in the graph.
   * @param colour array that the node colouring is copied into.
   */
  static void greedy(std::vector< std::deque<index_t> > &NNList, std::vector<bool> &active, index_t *colour){
    size_t NNodes = NNList.size();
    
    // Colour first active node.
    size_t node;
    for(node=0;node<NNodes;node++){
      if(active[node]){
        colour[node] = 0;
        break;
      }
    }
    
    // Colour remaining active nodes.
    for(;node<NNodes;node++){
      if(!active[node])
        continue;
      
      std::set<index_t> used_colours;
      for(typename std::deque<index_t>::const_iterator it=NNList[node].begin();it!=NNList[node].end();++it)
        if(*it<(int)node)
          used_colours.insert(colour[*it]);
      
      for(index_t i=0;;i++)
        if(used_colours.count(i)==0){
          colour[node] = i;
          break;
        }
    }
  }
};
#endif
