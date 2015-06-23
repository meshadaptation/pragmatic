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

#ifndef LOCK_H
#define LOCK_H

#include <atomic>

typedef unsigned char t_atomic;

class Lock
{
public:
    Lock() : _lock(0) {}

    Lock(const t_atomic value) : _lock(value) {}

    Lock(const Lock& other) :_lock(other._lock.load(std::memory_order_relaxed)) {}

    ~Lock() {}

    inline bool try_lock()
    {
        t_atomic lock_val = _lock.load(std::memory_order_relaxed);
        if(lock_val == 1)
            return false;
        lock_val = _lock.fetch_or(1, std::memory_order_acq_rel);

        return (lock_val==0);
    }

    inline void lock()
    {
        t_atomic lock_val = _lock.load(std::memory_order_relaxed);
        if(lock_val == 1) {
            while(!try_lock()) {};
            return;
        }

        bool is_unlocked = _lock.compare_exchange_weak(lock_val, 1, std::memory_order_acq_rel, std::memory_order_relaxed);
        if(!is_unlocked) {
            while(!try_lock()) {};
            return;
        }
    }

    inline bool is_locked()
    {
        return (_lock.load(std::memory_order_acquire) == 1);
    }

    inline void unlock()
    {
        _lock.store(0, std::memory_order_release);
    }

private:
    std::atomic<t_atomic> _lock;
};

#endif
