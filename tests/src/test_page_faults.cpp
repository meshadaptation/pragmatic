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

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef HAVE_LIBNUMA
#include <numaif.h>
#include <numa.h> 

void check_page_residence(void *array, size_t len, unsigned char *vec){
  if(mincore(array, len, vec)){
    if(errno==EAGAIN)
      perror("kernel is temporarily out of resources");
    else if(errno==EFAULT)
      perror("vec points to an invalid address.");
    else if(errno==EINVAL){
      perror("addr is not a multiple of the page size");
    }else if(errno==ENOMEM)
      perror("length is greater than (TASK_SIZE - addr).  (This could occur if a \
             negative value is specified for length, since that value will be \
             interpreted as a large unsigned integer.)  In Linux 2.6.11 and earlier, \
             the error EINVAL was returned for this condition.");
    else if(errno==ENOMEM)
      perror("addr to addr + length contained unmapped memory.");
    else if(errno)
      perror("unknown mincore() error");
  }
  return;
}
#endif
  
int main(){
#ifdef HAVE_LIBNUMA
  size_t npages = 13;
  size_t PAGE_SIZE = getpagesize();
  
  size_t length_of_array = npages*PAGE_SIZE/sizeof(double);
  size_t length_of_array_bytes = npages*PAGE_SIZE;
  
  std::vector<unsigned char> vec(npages);
  struct rusage usage;
  
  int nmemnodes = numa_max_node()+1;
  std::cout<<"Number of memory nodes = "<<nmemnodes<<std::endl;

  // std::vector<double> array(length_of_array);
  // double *array = new double[length_of_array];
  double *array=NULL;
  if(posix_memalign((void **)(&array), PAGE_SIZE, length_of_array_bytes))
    perror("posix_memalign");
  
  check_page_residence(array, length_of_array_bytes, (unsigned char *)(&(vec[0])));
  std::cout<<"Checking page residence\n";
  for(size_t i=0;i<npages;i++)
    std::cout<<"page "<<i<<" = "<<(int)vec[i]<<std::endl;
  
  getrusage(RUSAGE_SELF, &usage);
  int page_fault_pos0 = usage.ru_minflt;

  for(size_t i=0;i<length_of_array;i++)
    array[i] = i;

  getrusage(RUSAGE_SELF, &usage);
  int page_fault_pos1 = usage.ru_minflt;
  std::cout<<"Number of page faults after referencing memory = "
           <<page_fault_pos1-page_fault_pos0<<std::endl;

  check_page_residence(array, length_of_array_bytes, (unsigned char *)(&(vec[0])));
  std::cout<<"Re-checking page residence\n";
  for(size_t i=0;i<npages;i++)
    std::cout<<"page "<<i<<" = "<<(int)vec[i]<<std::endl;

  // Return the node ID of the node on which each page is allocated.
  for(size_t i=0;i<npages;i++){
    int mode;
    void *addr = (void *)(array + i*PAGE_SIZE/sizeof(double));
    unsigned long flags = MPOL_F_NODE|MPOL_F_ADDR;
    if(get_mempolicy(&mode, NULL, 0, addr, flags)){
      perror("get_mempolicy()");
    }
    std::cout<<"Page "<<i<<", node ID = "<<mode<<std::endl;
  }
#endif
  std::cout<<"pass\n";

  return 0;
}
