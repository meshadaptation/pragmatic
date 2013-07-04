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

#include "PragmaticTypes.h"

/*! \brief Performs a simple first breath greedy graph colouring of a local undirected graph.
 */
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
      for(std::vector<index_t>::const_iterator it=NNList[node].begin();it!=NNList[node].end();++it){
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

  /*! This routine colours a undirected graph using Gebremedhin &
   *  Manne's algorithm, "Scalable parallel graph coloring
   *  algorithms".
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void GebremedhinManne(std::vector< std::vector<index_t> > &NNList, int *colour){
    size_t NNodes = NNList.size();
    std::vector<bool> conflicts(NNodes, false);
#pragma omp parallel firstprivate(NNodes)
    {
      // Initialize.
#pragma omp for
      for(size_t i=0;i<NNodes;i++){
        colour[i] = 0;
      }

      // Phase 1: pseudo-colouring. Note - assuming graph can be colored with fewer than 64 colours.
#pragma omp for
      for(size_t i=0;i<NNodes;i++){
        unsigned long colours = 0;
        unsigned long c;
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
#pragma omp atomic read
          c = colour[*it];
          colours = colours | 1<<c;
        }
        colours = ~colours;

        for(unsigned int j=0;j<64;j++){
          c=1<<j;
          if(colours&c){
            colour[i] = j;
            break;
          }
        }
      }

      // Phase 2: find conflicts
#pragma omp for
      for(size_t i=0;i<NNodes;i++){
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
          conflicts[i] = conflicts[i] || (colour[i]==colour[*it]);
        }
      }

      // Phase 3: serial resolution of conflicts
      for(size_t i=0;i<NNodes;i++){
        if(!conflicts[i])
          continue;
        
        unsigned int colours = 0;
        int c;
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
          c = colour[*it];
          colours = colours | 1<<c;
        }
        colours = ~colours;

        for(unsigned int j=0;j<64;j++){
          c = 1<<j;
          if(colours&c){
            colour[i] = j;
            break;
          }else{
            c = c<<1;
          }
        }
      }
    }
  }

  /*! This routine repairs the colouring - based on the second and
   *  third phases of Gebremedhin & Manne's algorithm, "Scalable
   *  parallel graph coloring algorithms".
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void repair(std::vector< std::vector<index_t> > &NNList, int *colour){
    size_t NNodes = NNList.size();
    std::vector<bool> conflicts(NNodes, false);
#pragma omp parallel firstprivate(NNodes)
    {
      // Phase 2: find conflicts
#pragma omp for
      for(size_t i=0;i<NNodes;i++){
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
          conflicts[i] = conflicts[i] || (colour[i]==colour[*it]);
        }
      }

      // Phase 3: serial resolution of conflicts
      for(size_t i=0;i<NNodes;i++){
        if(!conflicts[i])
          continue;
        
        unsigned int colours = 0;
        int c;
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
          c = colour[*it];
          colours = colours | 1<<c;
        }
        colours = ~colours;

        for(unsigned int j=0;j<64;j++){
          c = 1<<j;
          if(colours&c){
            colour[i] = j;
            break;
          }else{
            c = c<<1;
          }
        }
      }
    }
  }

};
#endif
