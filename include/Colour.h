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

#include <vector>

#include "PragmaticTypes.h"

/*! \brief Performs a simple first breath greedy graph colouring of a local undirected graph.
 */
class Colour{
 public:
  /*! This routine colours a undirected graph using the greedy colouring algorithm.
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void greedy(size_t NNodes, std::vector< std::vector<index_t> > &NNList, std::vector<char> &colour){
    char max_colour=64;

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
          if(colour[*it]>=max_colour){
            max_colour*=2;
            used_colours.resize(max_colour, false);
          }

          used_colours[colour[*it]] = true;
        }
      }

      for(char i=0;;i++)
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
  static void GebremedhinManne(size_t NNodes, std::vector< std::vector<index_t> > &NNList, std::vector<char> &colour){
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
        char c;
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
#pragma omp atomic read
          c = colour[*it];
          colours = colours | 1<<c;
        }
        colours = ~colours;

        for(size_t j=0;j<64;j++){
          if(colours&(1<<j)){
            colour[i] = j;
            break;
          }
        }
      }

      // Phase 2: find conflicts
      std::vector<size_t> conflicts;
#pragma omp for
      for(size_t i=0;i<NNodes;i++){
        for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
          if(colour[i]==colour[*it]){
            conflicts.push_back(i);
            break;
          }
        }
      }

      // Phase 3: serial resolution of conflicts
      int tid = pragmatic_thread_id();
      int nthreads = pragmatic_nthreads();
      for(int i=0;i<nthreads;i++){
        if(tid==i){  
          for(std::vector<size_t>::const_iterator it=conflicts.begin();it!=conflicts.end();++it){
            unsigned long colours = 0;
            for(std::vector<index_t>::const_iterator jt=NNList[*it].begin();jt!=NNList[*it].end();++jt){
              colours = colours | 1<<(colour[*jt]);
            }
            colours = ~colours;

            for(size_t j=0;j<64;j++){
              if(colours&(1<<j)){
                colour[*it] = j;
                break;
              }
            }
          }
        }
#pragma omp barrier
      }
    }
  }

  /*! This routine repairs the colouring - based on the second and
   *  third phases of Gebremedhin & Manne's algorithm, "Scalable
   *  parallel graph coloring algorithms".
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void repair(size_t NNodes, std::vector< std::vector<index_t> > &NNList, std::vector<char> &colour){
    // Phase 2: find conflicts
    std::vector<size_t> conflicts;
#pragma omp for
    for(size_t i=0;i<NNodes;i++){
      for(std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
        if(colour[i]==colour[*it]){
          conflicts.push_back(i);
          break;
        }
      }
    }
    
    // Phase 3: serial resolution of conflicts
    int tid = pragmatic_thread_id();
    int nthreads = pragmatic_nthreads();
    for(int i=0;i<nthreads;i++){
      if(tid==i){  
        for(std::vector<size_t>::const_iterator it=conflicts.begin();it!=conflicts.end();++it){
          unsigned long colours = 0;
          for(std::vector<index_t>::const_iterator jt=NNList[*it].begin();jt!=NNList[*it].end();++jt){
            colours = colours | 1<<(colour[*jt]);
          }
          colours = ~colours;
          
          for(size_t j=0;j<64;j++){
            if(colours&(1<<j)){
              colour[*it] = j;
              break;
            }
          }
        }
      }
#pragma omp barrier
    }
  }

};
#endif
