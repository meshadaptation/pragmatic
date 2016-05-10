/*  Copyright (C) 2015 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Georgios Rokos
 *  Software Performance Optimisation Group
 *  Department of Computing
 *  Imperial College London
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

#ifndef WORK_STEALING_QUEUE_H
#define WORK_STEALING_QUEUE_H

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "Worklist.h"

template<typename t_type>
class WorkStealingQueue
{
public:
    WorkStealingQueue()
    {
        wsq.resize(omp_get_max_threads());
    }

    inline void push(t_type value, int tid)
    {
        wsq[tid].push(value);
    }

    inline t_type pop(int tid)
    {
        return wsq[tid].pop();
    }

private:
    struct ThreadQueue {
        ThreadQueue() : _current(&_worklist_odd), _next(&_worklist_even), _empty(true)
        {
            _current->init_creation();
            _next->init_creation();
        }

        inline void push(t_type value)
        {
            _next->push_back(value);
            _empty = false;
        }

        inline t_type pop()
        {
            // Get the next workitem...
            t_type next_item = _current->get_next();

            /*
             * ...and before returning take care of special situations:
             * 1) if current queue is now empty, swap current and next
             * 2) if both queues are empty, attempt to steal work
             */
            if(!_current->is_valid()) {
                if(_next->is_valid()) {
                    // Case 1; swap worklists
                    _lock.lock();
                    Worklist<t_type> * tmp = _current;
                    _current = _next;
                    _next = tmp;
                    _current->init_traversal();
                    _next->init_creation();
                    _lock.unlock();
                } else {
                    // Case 2; we need to steal work
                    _empty = true;
                    int tid = omp_get_thread_num();
                    int nthreads = omp_get_max_threads();
                    for(int i = (tid+1)%nthreads; i != tid; i = (i+1)%nthreads) {
                        if(wsq[i]._next->steal_work(*_current)) {
                            _empty = false;
                            break;
                        }
                    }
                }
            }

            return next_item;
        }

        inline bool is_empty()
        {
            return _empty;
        }

        Worklist<t_type> _worklist_odd, _worklist_even;
        Worklist<t_type> * _current, * _next;
        bool _empty;

        // Lock used for critical operations (work stealing, list swapping)
        Lock _lock;
    };

    std::vector<ThreadQueue> wsq;
};

#endif
