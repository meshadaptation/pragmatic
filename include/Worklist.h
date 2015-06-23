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

#ifndef WORKLIST_H
#define WORKLIST_H

#include <cassert>
#include <cstdlib>
#include <vector>

#include "Lock.h"

template<typename t_type>
class Worklist
{
private:
    template<typename t_type> friend class WorkStealingQueue;

    Worklist() : _mode(INVALID), _lock(0)
    {
        _init();
    }

    Worklist(const Worklist& other)
        : _list(other._list), _reserved(other._reserved),
          _end_idx(other._end_idx), _lock(other._lock),
          _mode(other._mode.load(std::memory_order_relaxed)),
          _idx(other._idx.load(std::memory_order_relaxed)),
          _first_idx(other._first_idx.load(std::memory_order_relaxed))
    {
        _init();
    }

    ~Worklist()
    {
        free(_list);
    }

    void init_creation()
    {
        _idx.store(0, std::memory_order_relaxed);
        _first_idx.store(0, std::memory_order_relaxed);
        _mode.store(CREATION, std::memory_order_release);
    }

    inline void push_back(t_type value)
    {
        assert(_mode.load(std::memory_order_relaxed)==CREATION);
        long int next_idx = _idx.fetch_add(1, std::memory_order_relaxed);
        if(next_idx >= _end_idx) {
            _expand();
        }
        _list[next_idx] = value;
    }

    void init_traversal()
    {
        --_idx; // Place _idx to the first item available for consumption
        _mode.store(TRAVERSAL, std::memory_order_release);
    }

    inline t_type get_next()
    {
        assert(_mode.load(std::memory_order_relaxed)==TRAVERSAL);
        long int next_idx = _idx.fetch_sub(1, std::memory_order_relaxed);
        if(next_idx == _first_idx)
            _mode.store(INVALID, std::memory_order_release);
        return _list[next_idx];
    }

    inline bool is_valid()
    {
        return _mode.load(std::memory_order_relaxed) != INVALID;
    }

    bool steal_work(Worklist<t_type>& thiefs_worklist)
    {
        if(_mode.load(std::memory_order_acquire) == INVALID)
            return false;

        long int current, target;
        bool success = false;
        _lock.lock(); // protect _list from being swapped while we are stealing its contents
        do {
            current = _idx.load(std::memory_order_relaxed);
            if(current < 10)
                break;

            target = current >> 1;
            success = _idx.compare_exchange_weak(current, target, std::memory_order_relaxed, std::memory_order_relaxed);
        } while(!success);

        if(success) {
            long int nwork = current - target;
            if(thiefs_worklist._end_idx < nwork) {
                thiefs_worklist._expand();
            }
            thiefs_worklist._idx = nwork;
            std::memcpy(thiefs_worklist._list, &_list[target], nwork*sizeof(t_type));
        }
        _lock.unlock();

        return success;
    }

    void _init()
    {
        _reserved = 1 << 12; // Allocate a page of memory (4k).
        void* alloc = std::malloc(_reserved);
        if(alloc == NULL) {
            std::perror("Bad alloc");
            std::exit(1);
        }
        _list = static_cast<t_type *>(alloc);
        _end_idx = _reserved / sizeof(t_type);
    }

    void _expand()
    {
        _reserved <<= 1; // Allocate twice as much
        _lock.lock();
        void* alloc = std::realloc(_list, _reserved);
        if(alloc == NULL) {
            std::perror("Bad alloc");
            std::exit(1);
        }
        _list = static_cast<t_type *>(alloc);
        _lock.unlock();
        _end_idx = _reserved / sizeof(t_type);
    }

    enum Mode {CREATION, TRAVERSAL, INVALID};

    // Array containing the worklist
    t_type* _list;

    // Amount of bytes reserved
    size_t _reserved;

    // The first out-of-bounds index
    long int _end_idx;

    // Current mode, i.e. whether the worklist is being populated with
    // workitems or whether workitems from the list are being processed
    std::atomic<Mode> _mode;

    // First position in _list which is available for storing the next workitem if
    // in creation mode or index to the first unread workitem if in traversal mode.
    std::atomic<long int> _idx;

    // Index to the first workitem in _list which is available for consumption.
    std::atomic<long int> _first_idx;

    // Lock used for critical operations, i.e. memory reallocation
    Lock _lock;
};
#endif
