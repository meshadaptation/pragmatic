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

#ifndef H_EDGE
#define H_EDGE

/*! \brief Mesh edge object.
*/
template<typename index_t> class Edge
{
public:
    /*! Constructor
     * @param nid0 Node-Id 0
     * @param nid1 Node-Id 1
     */
    Edge(index_t nid0, index_t nid1)
    {
        edge.first = std::min(nid0, nid1);
        edge.second = std::max(nid0, nid1);
    }

    /*! Copy constructor
     * @param in Edge object.
     */
    Edge(const Edge& in)
    {
        *this = in;
    }

    /// Destructor
    ~Edge() {}

    /// Assignment operator
    Edge& operator=(const Edge &in)
    {
        edge = in.edge;

        return *this;
    }

    /// Equality operator.
    bool operator==(const Edge& in) const
    {
        return this->edge == in.edge;
    }

    /// Inequality operator.
    bool operator!=(const Edge& in) const
    {
        return this->edge != in.edge;
    }

    /// Less-than operator
    bool operator<(const Edge& in) const
    {
        return this->edge < in.edge;
    }

    index_t connected(const Edge& in) const
    {
        if((edge.first==in.edge.first)||(edge.first==in.edge.second))
            return edge.first;
        else if((edge.second==in.edge.first)||(edge.second==in.edge.second))
            return edge.second;
        return -1;
    }

    bool contains(index_t nid) const
    {
        return (nid==edge.first)||(nid==edge.second);
    }

    template<typename _real_t> friend class Mesh;
    template<typename _real_t, int _dim> friend class Coarsen;
    template<typename _real_t, int _dim> friend class Swapping;
    template<typename _real_t, int _dim> friend class Refine;

private:

    std::pair<index_t, index_t> edge;
};

template<typename index_t> class DirectedEdge
{
public:
    /*! Constructor
     * @param nid0 Node-Id 0
     * @param nid1 Node-Id 1
     */
    DirectedEdge(index_t nid0, index_t nid1)
    {
        edge.first = nid0;
        edge.second = nid1;
    }

    DirectedEdge(index_t nid0, index_t nid1, index_t nid)
    {
        edge.first = nid0;
        edge.second = nid1;
        id = nid;
    }

    /*! Copy constructor
     * @param in DirectedEdge object.
     */
    DirectedEdge(const DirectedEdge& in)
    {
        *this = in;
    }

    // Default constructor.
    DirectedEdge() {}

    /// Destructor
    ~DirectedEdge() {}

    /// Assignment operator
    DirectedEdge& operator=(const DirectedEdge &in)
    {
        edge = in.edge;
        id = in.id;

        return *this;
    }

    /// Equality operator.
    bool operator==(const DirectedEdge& in) const
    {
        return this->edge == in.edge;
    }

    /// Inequality operator.
    bool operator!=(const DirectedEdge& in) const
    {
        return this->edge != in.edge;
    }

    /// Less-than operator
    bool operator<(const DirectedEdge& in) const
    {
        return this->edge < in.edge;
    }

    index_t connected(const DirectedEdge& in) const
    {
        if((edge.first==in.edge.first)||(edge.first==in.edge.second))
            return edge.first;
        else if((edge.second==in.edge.first)||(edge.second==in.edge.second))
            return edge.second;
        return -1;
    }

    bool contains(index_t nid) const
    {
        return (nid==edge.first)||(nid==edge.second);
    }

    template<typename _real_t> friend class Mesh;
    template<typename _real_t, int _dim> friend class Coarsen;
    template<typename _real_t, int _dim> friend class Swapping;
    template<typename _real_t, int _dim> friend class Refine;

private:
    index_t id;
    std::pair<index_t, index_t> edge;
};

#endif

