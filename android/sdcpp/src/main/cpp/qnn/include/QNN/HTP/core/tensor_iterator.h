// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef HEXNN_TENSOR_ITERATOR_H
#define HEXNN_TENSOR_ITERATOR_H 1

#include <cstddef>
#include <array>

#include "tensor.h"

PUSH_VISIBILITY(default)

template <typename T> class TensorIter;
template <typename T> class TensorCIter;

template <typename T> class IterableTensor {
    typedef TensorIter<T> iterator;
    typedef TensorCIter<T> const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;

  protected:
    pointer myTensor;
    const_pointer myCTensor;
    const std::array<size_t, 4> increments;
    mutable std::array<size_t, 4> dims;
    const bool is_const;

  public:
    friend iterator; //class TensorIter<T> ;
    friend const_iterator;

    API_EXPORT inline IterableTensor(reference t, std::array<size_t, 4> inc)
        : myTensor(&t), myCTensor(const_cast<const_pointer>(&t)), increments(inc), is_const(false)
    {
        assert(myCTensor && myCTensor->rank() == 4);
        for (int i = 0; i < 4; i++) {
            dims[i] = myCTensor->dim(i);
        }
    }

    API_EXPORT inline IterableTensor(const_reference t, std::array<size_t, 4> inc)
        : myTensor(nullptr), myCTensor(&t), increments(inc), is_const(true)
    {
        assert(myCTensor && myCTensor->rank() == 4);
        for (int i = 0; i < 4; i++) {
            dims[i] = myCTensor->dim(i);
        }
    }
    IterableTensor(IterableTensor const &) = delete;
    IterableTensor(IterableTensor &&) = delete;
    IterableTensor &operator=(IterableTensor const &) = delete;
    IterableTensor &operator=(IterableTensor &&) = delete;

    API_EXPORT inline size_t dim(size_t index) const { return dims[index]; }

    API_EXPORT inline auto access(size_t b, size_t h, size_t w, size_t d) &
    {
        assert(!is_const && myTensor);
        return (*myTensor)(b, h, w, d);
    }
    API_EXPORT inline auto access(size_t b, size_t h, size_t w, size_t d) &&
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }
    API_EXPORT inline auto access(size_t b, size_t h, size_t w, size_t d) const &&
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }

    API_EXPORT inline auto read(size_t b, size_t h, size_t w, size_t d) const
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }

    API_EXPORT inline auto operator()(size_t b, size_t h, size_t w, size_t d) { return access(b, h, w, d); }
    API_EXPORT inline auto operator()(size_t b, size_t h, size_t w, size_t d) const
    {
        assert(myCTensor);
        return (*myCTensor)(b, h, w, d);
    }

    API_EXPORT inline bool operator==(const IterableTensor<T> &it) const
    {
        return (this->myCTensor == it.myCTensor) && (this->increments == it.increments);
    }
    API_EXPORT inline bool operator!=(const IterableTensor<T> &it) const { return !(*this == it); }

    API_EXPORT inline iterator begin()
    {
        std::array<size_t, 4> const start = {0, 0, 0, 0};
        return iterator(*this, start);
    }
    API_EXPORT inline const_iterator begin() const
    {
        std::array<size_t, 4> start = {0, 0, 0, 0};
        return const_iterator(*this, start);
    }
    API_EXPORT inline iterator begin(std::array<size_t, 4> start) { return iterator(*this, start); }
    API_EXPORT inline const_iterator begin(std::array<size_t, 4> start) const { return const_iterator(*this, start); }
    API_EXPORT inline iterator end()
    {
        std::array<size_t, 4> const end = {dims[0], 0, 0, 0};
        return iterator(*this, end);
    }
    API_EXPORT inline const_iterator end() const
    {
        std::array<size_t, 4> end = {dims[0], 0, 0, 0};
        return const_iterator(*this, end);
    }
    API_EXPORT inline iterator end(std::array<size_t, 4> end) { return iterator(*this, end); }
    API_EXPORT inline const_iterator end(std::array<size_t, 4> end) const { return const_iterator(*this, end); }

    API_EXPORT ~IterableTensor() {}
};

template <typename T> class TensorIter {
  private:
    IterableTensor<T> &myITensor;
    std::array<size_t, 4> location;
    API_EXPORT bool increment(size_t dim)
    {
        size_t const inc = myITensor.increments[dim];
        if (inc) {
            size_t const loc = location[dim];
            if (loc + inc < myITensor.dim(dim)) {
                location[dim] += inc;
                return true;
            } else if (dim != 0) {
                location[dim] = 0;
            }
        }
        if (dim == 0) {
            location[0]++;
            return true;
        }

        return false;
    }

    API_EXPORT inline void incrementLocation()
    {
        int i = location.size();
        while (!increment(--i))
            ;
    }

  protected:
    API_EXPORT inline TensorIter(const TensorIter<T> &to_copy)
        : myITensor(to_copy.myITensor), location(to_copy.location)
    {
    }
    TensorIter(TensorIter &&) = delete;
    TensorIter &operator=(TensorIter const &) = delete;
    TensorIter &operator=(TensorIter &&) = delete;

  public:
    API_EXPORT inline TensorIter(IterableTensor<T> &it, std::array<size_t, 4> loc) : myITensor(it), location(loc) {}

    API_EXPORT inline TensorIter<T> &clone() { return TensorIter<T>(*this); }

    API_EXPORT inline std::array<size_t, 4> get_location() { return location; }

    API_EXPORT inline bool operator==(const TensorIter<T> &ti) const
    {
        return this->myITensor == ti.myITensor && this->location == ti.location;
    }
    API_EXPORT inline bool operator!=(const TensorIter<T> &ti) const { return !(*this == ti); }
    API_EXPORT inline operator float() const { return myITensor(location[0], location[1], location[2], location[3]); }
    API_EXPORT inline TensorIter<T> &operator=(const float v)
    {
        myITensor(location[0], location[1], location[2], location[3]) = v;
        return *this;
    }
    //inline TensorIter<T>&operator=(const TensorIter<T>& v) { return this->operator=(float(v)); }
    //inline auto & operator*() {return myITensor(location[0],location[1],location[2],location[3]);}
    API_EXPORT inline TensorIter<T> &operator++()
    {
        incrementLocation();
        return *this;
    }
    API_EXPORT inline TensorIter<T> operator++(int)
    {
        TensorIter<T> const clone = TensorIter<T>(*this);
        incrementLocation();
        return clone;
    }

    ~TensorIter() {}
};

template <typename T> class TensorCIter {
  private:
    const IterableTensor<T> &myITensor;
    std::array<size_t, 4> location;
    API_EXPORT bool increment(size_t dim)
    {
        const size_t inc = myITensor.increments[dim];
        if (inc) {
            size_t const loc = location[dim];
            if (loc + inc < myITensor.dim(dim)) {
                location[dim] += inc;
                return true;
            } else if (dim != 0) {
                location[dim] = 0;
            }
        }
        if (dim == 0) {
            location[0]++;
            return true;
        }

        return false;
    }

    API_EXPORT inline void incrementLocation()
    {
        int i = location.size();
        while (!increment(--i))
            ;
    }

  public:
    API_EXPORT inline TensorCIter(const IterableTensor<T> &it, std::array<size_t, 4> loc) : myITensor(it), location(loc)
    {
    }
    TensorCIter(TensorCIter const &) = default;
    TensorCIter(TensorCIter &&) = delete;
    TensorCIter &operator=(TensorCIter const &) = delete;
    TensorCIter &operator=(TensorCIter &&) = delete;

    API_EXPORT inline std::array<size_t, 4> get_location() { return location; }

    API_EXPORT inline bool operator==(const TensorCIter<T> &ti) const
    {
        return this->myITensor == ti.myITensor && this->location == ti.location;
    }
    API_EXPORT inline bool operator!=(const TensorCIter<T> &ti) const { return !(*this == ti); }
    API_EXPORT inline operator float() const
    {
        return myITensor.read(location[0], location[1], location[2], location[3]);
    }
    API_EXPORT inline TensorCIter<T> &operator++()
    {
        incrementLocation();
        return *this;
    }
    API_EXPORT inline TensorCIter<T> operator++(int)
    {
        TensorCIter<T> clone(*this);
        incrementLocation();
        return clone;
    }

    ~TensorCIter() {}
};

POP_VISIBILITY()

#endif // HEXNN_TENSOR_ITERATOR_H
