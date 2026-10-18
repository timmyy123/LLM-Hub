// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_INTERFACE_H
#define HEXNN_INTERFACE_H 1

#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include <type_traits>

#include "conversions.h"
#include "forward_classes.h"
#include "macros_attribute.h"
#include "deserializer.h"
#include "float16.h"
#include "dtype.h"

PUSH_VISIBILITY(default)

namespace hnnx {
class InterfaceRef;
struct intfc_methods;
} // namespace hnnx

//////// dtype_of_type<INTFC> maps interface class to DType.
// the general def is in 'dtype.h' and maps to 'UNKNOWN'.
class NullInterface;
template <typename T> class PlainInterface;
template <typename T> class ScaleOffsetInterface;

template <> constexpr DType dtype_of_type<ScaleOffsetInterface<uint8_t>>()
{
    return DType::QUInt8;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<int8_t>>()
{
    return DType::QInt8;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<uint16_t>>()
{
    return DType::QUInt16;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<int16_t>>()
{
    return DType::QInt16;
}

template <> constexpr DType dtype_of_type<PlainInterface<Float16>>()
{
    return DType::Float16;
}
template <> constexpr DType dtype_of_type<PlainInterface<BFloat16>>()
{
    return DType::BFloat16;
}
template <> constexpr DType dtype_of_type<PlainInterface<float>>()
{
    return DType::Float32;
}
template <> constexpr DType dtype_of_type<ScaleOffsetInterface<NN_INT32_T>>()
{
    return DType::QInt32;
}
template <> constexpr DType dtype_of_type<PlainInterface<NN_INT32_T>>()
{
    return DType::Int32;
}
template <> constexpr DType dtype_of_type<PlainInterface<NN_INT64_T>>()
{
    return DType::Int64;
}

/*
 * An Interface has all the necessary values and functionality to encode and decode values
 *
 * virtual methods do generic conversion to/from floats, with a void * to the encoded data.
 *
 * Each concrete Tensor (and some less-than-concrete) has an instance of an Interface.
 *
 * IMPORTANY: All interface classes must be trivially destructible.
 * As a result, even though we have virtual methods, it is safe to
 * have no virtual dtor.
 * This is important for performance, since every tensor has an Interface subclass
 * embedded in it; and most tensor classes need no destructor for any other reason.
 * So a dtor requirement in the 'interface' could add time to the teardown,
 * even if the dtors don't do very much.
 */

class Interface {
  protected:
    // base class has the 'dtype info' for the interface. It must occupy an
    // aligned 4-byte location.
    alignas(4) dtype_info dtinfo;

    explicit constexpr Interface(dtype_info const dti) : dtinfo(dti) {}
    using intfc_methods = hnnx::intfc_methods;

  public:
    // Base class can read this info directly from the 'dtinfo' field:
    constexpr dtype_info get_dt_info() const noexcept { return dtinfo; }
    constexpr unsigned element_size() const noexcept { return dtinfo.elbytes; }
    constexpr DType get_dtype() const noexcept { return dtinfo.dtype; }
    constexpr bool is_quantized() const noexcept { return dtinfo.is_quant; }

    struct qparms {
        int offset;
        float scale;
        float scale_recip;
    };
    using read_float_fp = float (*)(Interface const *, void const *) noexcept;
    using write_float_fp = void (*)(Interface const *, void *, const float) noexcept;
    using get_qparms_fp = qparms const *(*)(Interface const *) noexcept;
    using ifc_hash_fp = uint32_t (*)(Interface const *) noexcept;
    using ifc_compare_fp = int (*)(Interface const *, Interface const *) noexcept;
    static unsigned constexpr N_types = unsigned(DType::ZZ_LAST_DTYPE);

    // This constructs an InterfaceRef for an arbitrary Interface instance, by using the dtype
    // in the header word to select the method table.
    // This is in tensor.cc; it is expected to be used fairly rarely, and maybe not at all at runtime,
    // but it will be pretty quick.
    API_EXPORT hnnx::InterfaceRef get_refobj() const;

    template <typename IFC> static Interface const *canonical_instance_for(); // inline is below

  protected:
    template <DType DT> static intfc_methods const &methods_for(); // inline is below

    API_EXPORT static constexpr qparms null_parms = {0, 1.0f, 1.0f};
    static qparms const *get_null_qparms(Interface const *) noexcept { return &null_parms; }
};

namespace hnnx {
// ONLY for use in intfc_methods
class IfcExemplar final : public Interface {
  public:
    constexpr IfcExemplar(dtype_info const dt) : Interface(dt) {}
    constexpr IfcExemplar() : Interface(dtype_info{}) {}
};

//
// for each 'concrete' subclass of Interace, there is one private instance
// of this, which is 'static constexpr methods_instance',
// e.g. PlainInterface<float>::methods_instance is one of these.
// The ifc_hash' method does not compute the complete hash; it remains to 'xor' with unsigned(dtype).
// In cases where it just returns 0, a null ptr can be used.
//
struct intfc_methods {
    hnnx::IfcExemplar exemplar; // <- contains the dtype_info
    Interface::read_float_fp read_float;
    Interface::write_float_fp write_float;
    Interface::get_qparms_fp get_qparms;
    Interface::ifc_hash_fp ifc_hash; // <- may be null if it always returns 0 (e.g. PlainInterface)
    Interface::ifc_compare_fp ifc_compare; // <- may be null if always returns 0 (e.g. PlainIterface).
};
// All of the intfc_methods are stored in this table, which can be indexed by DType.
using ifc_method_table_t = std::array<intfc_methods, Interface::N_types>;
constexpr ifc_method_table_t construct_ifc_method_table(); // in tensor.cc; not publicly visible
// This is defined in tensor.cc, and built at compile time.
API_EXPORT_IMPORT extern const ifc_method_table_t ifc_method_table;

} // namespace hnnx
// can define this now.
template <DType DT> inline hnnx::intfc_methods const &Interface::methods_for()
{
    return hnnx::ifc_method_table[unsigned(DT)];
}
//
// for a given actual interface class, get a pointer to a 'canonical'
// instance; this is actually the 'exemplar' field in the methods table.
//  - Must be a subclass of Interface
//  - Must have a 'dtype' attribute (or be NullInterface)
//  - Must be the same size as Interface (cannot be used for ScaleOffsetInterface).
//
template <typename IFC> inline Interface const *Interface::canonical_instance_for()
{
    static_assert(std::is_base_of_v<Interface, IFC>);
    static_assert(sizeof(IFC) == sizeof(Interface));
    constexpr DType dt = IFC::dtype;
    return &hnnx::ifc_method_table[unsigned(dt)].exemplar;
}
// NullInterface doesn't have a dtype (and maybe shouldn't...)
template <> inline Interface const *Interface::canonical_instance_for<NullInterface>()
{
    return &hnnx::ifc_method_table[unsigned(DType::UNKNOWN)].exemplar;
}

namespace hnnx {
// The 'interface(); virtual method of Tensor now returns this object.
// If the interface is anything but ScaleOffsetInterface, the 'interface' ptr can be null.
class InterfaceRef {

  public:
    using qparms = Interface::qparms;

  protected:
    intfc_methods const *methods_p;
    Interface const *intfc_p;

  private: // use make_null() method if needed
    constexpr InterfaceRef() : methods_p(nullptr), intfc_p(nullptr) {}

  public:
    InterfaceRef(InterfaceRef const &) = default;
    InterfaceRef(InterfaceRef &&) = default;
    InterfaceRef &operator=(InterfaceRef const &) = default;
    InterfaceRef &operator=(InterfaceRef &&) = default;
    ~InterfaceRef() = default;

    API_EXPORT InterfaceRef(intfc_methods const &mthods, Interface const *ifc_p) : methods_p(&mthods), intfc_p(ifc_p) {}
    API_EXPORT qparms const *get_qparms() const { return methods_p->get_qparms(intfc_p); }
    API_EXPORT float get_scale() const { return methods_p->get_qparms(intfc_p)->scale; }
    API_EXPORT float get_scale_recip() const { return methods_p->get_qparms(intfc_p)->scale_recip; }
    API_EXPORT int32_t get_offset() const { return methods_p->get_qparms(intfc_p)->offset; }
    API_EXPORT void write_float(void *ptr, const float in) const noexcept { methods_p->write_float(intfc_p, ptr, in); }
    API_EXPORT float read_float(const void *ptr) const noexcept { return methods_p->read_float(intfc_p, ptr); }
    API_EXPORT uint32_t interface_hash() const noexcept
    {
        uint32_t h = methods_p->ifc_hash ? methods_p->ifc_hash(intfc_p) : 0;
        return h ^ uint32_t(methods_p->exemplar.get_dtype());
    }
    API_EXPORT dtype_info get_dt_info() const noexcept { return methods_p->exemplar.get_dt_info(); }
    API_EXPORT unsigned element_size() const noexcept { return methods_p->exemplar.element_size(); }
    API_EXPORT DType get_dtype() const noexcept { return methods_p->exemplar.get_dtype(); }
    API_EXPORT bool is_quantized() const noexcept { return methods_p->exemplar.is_quantized(); }
    // might as well have get_refobj() method, for consistency of interface...
    API_EXPORT InterfaceRef get_refobj() const { return *this; }

    // this is used as a 'pseudo-ctor' to make a null InterfaceRef, in a few places.
    // (null ctor is currently private)
    static constexpr InterfaceRef make_null() { return InterfaceRef{}; }

    Interface const *get_intfc_ptr() const { return intfc_p; }
    intfc_methods const *get_methods_ptr() const { return methods_p; }

    // Ordered compare of two 'InterfaceRef'. If the types are different (detected by
    // different method pointers), then we order based on the addresses of the method
    // tables; since the method tables are all in one big array, this means they are
    // ordered according to 'dtype'.
    API_EXPORT int compare(InterfaceRef const &rhs) const noexcept
    {
        if (methods_p != rhs.methods_p) return (methods_p < rhs.methods_p) ? -1 : 1;
        if (intfc_p == rhs.intfc_p) return 0; // same type, same object
        auto fp = methods_p->ifc_compare;
        return (fp == nullptr) ? 0 : (*fp)(intfc_p, rhs.intfc_p);
    }
    API_EXPORT bool compare_eq(InterfaceRef const &rhs) const noexcept
    {
        if (methods_p != rhs.methods_p) return false;
        if (intfc_p == rhs.intfc_p) return true; // same type, same object
        auto fp = methods_p->ifc_compare;
        return (fp == nullptr) ? true : ((*fp)(intfc_p, rhs.intfc_p) == 0);
    }
    friend bool operator==(InterfaceRef const &lhs, InterfaceRef const &rhs) noexcept { return lhs.compare_eq(rhs); }
    friend bool operator!=(InterfaceRef const &lhs, InterfaceRef const &rhs) noexcept { return !lhs.compare_eq(rhs); }
    friend bool operator<(InterfaceRef const &lhs, InterfaceRef const &rhs) noexcept { return lhs.compare(rhs) < 0; }
};

template <typename IFCT> Interface::qparms &qparms_for_interface_patch(size_t);
} // namespace hnnx

namespace hnnx {
// make_interface<INTFC>::from_odef( Graph &, OutputDef const &odef)
// returns a pointer to an INTFC suitable for odef, either
// by finding an existing one, or by adding a new one to the crate.
// make_interface<INTFC>::from_deser(Deseralizer & dctx)
// returns a pointer to an INTFC , deserialized,
// by finding an existing one which matches, or by adding a new one to the crate.
template <typename INTFC> struct make_interface {
    API_EXPORT static Interface const *from_odef(Graph &, OutputDef const &odef);
    API_EXPORT static Interface const *from_deser(Deserz &dctx, Interface const **ptrloc);
};
} // namespace hnnx

/*
 * But guess what... you can't ever instantiate an abstract class!
 * So if we want to return a generic Accessor, we need to make it non-abstract.
 *
 * We need an abstract pointer to some element.  The way to do this is void *
 * We need a pointer to the Interface, which needs to be able to work with a void *
 *
 * This pushes the runtime polymorphism into the Interface, which we can share
 * between Accessor instances
 */

class GenericAccessorRO {
  protected:
    void *data;
    const Interface &interface;
    Interface::read_float_fp read_fp;

  public:
    ALWAYSINLINE GenericAccessorRO(void const *const data_in, hnnx::InterfaceRef const &interface_in)
        : data(const_cast<void *>(data_in)), interface(*interface_in.get_intfc_ptr()),
          read_fp(interface_in.get_methods_ptr()->read_float)
    {
    }
    ALWAYSINLINE GenericAccessorRO(GenericAccessorRO const &) = default;
    ALWAYSINLINE GenericAccessorRO(GenericAccessorRO &&) = default;
    GenericAccessorRO &operator=(GenericAccessorRO const &) = delete;
    GenericAccessorRO &operator=(GenericAccessorRO &&) = delete;
    ALWAYSINLINE ~GenericAccessorRO() = default;

    typedef GenericAccessorRO AccessorRO;
    ALWAYSINLINE float as_float() const { return (*read_fp)(&interface, data); }
    ALWAYSINLINE operator float() const { return as_float(); }
};
class GenericAccessor : public GenericAccessorRO {
    Interface::write_float_fp write_fp;

  public:
    ALWAYSINLINE GenericAccessor(void *const data_in, hnnx::InterfaceRef const &interface_in)
        : GenericAccessorRO(data_in, interface_in), write_fp(interface_in.get_methods_ptr()->write_float)
    {
    }
    ALWAYSINLINE GenericAccessor(GenericAccessor const &) = default;
    ALWAYSINLINE GenericAccessor(GenericAccessor &&) = default;
    ALWAYSINLINE ~GenericAccessor() = default;

    ALWAYSINLINE void set_float(float v) { write_fp(&interface, data, v); }
    ALWAYSINLINE float operator=(float v)
    {
        set_float(v);
        return v;
    }
    ALWAYSINLINE float operator=(GenericAccessorRO const &rhs)
    {
        float const v = rhs.as_float();
        set_float(v);
        return v;
    }
    ALWAYSINLINE float operator=(GenericAccessor const &rhs)
    {
        if (this != &rhs) {
            return operator=(static_cast<GenericAccessorRO const &>(rhs));
        }
        return this->as_float();
    }
    // only needed for SA; forward to 'const &' operator=
    ALWAYSINLINE float operator=(GenericAccessor &&rhs) { return this->operator=(rhs); }
};
// this is returned by Tensor::get_dtype_intfc()
//
struct DTypeScaleOff {
    DType dtype;
    float scale;
    int offset;
    DTypeScaleOff(DType dt, float sc, int zo) noexcept : dtype(dt), scale(sc), offset(zo) {}
    explicit DTypeScaleOff(DType dt) noexcept : dtype(dt), scale(1.0f), offset(0) {}
    DTypeScaleOff() noexcept : DTypeScaleOff(DType::UNKNOWN) {}
    // construct from dtype and qparms ref, etc...
    DTypeScaleOff(DType dt, Interface::qparms const &qpp) noexcept : dtype(dt), scale(qpp.scale), offset(qpp.offset) {}
    DTypeScaleOff(DType dt, hnnx::InterfaceRef const &iref) noexcept : DTypeScaleOff(dt, *iref.get_qparms()) {}
    explicit DTypeScaleOff(hnnx::InterfaceRef const &iref) noexcept
        : DTypeScaleOff(iref.get_dtype(), *iref.get_qparms())
    {
    }
    DTypeScaleOff(DTypeScaleOff const &) = default;
    DTypeScaleOff(DTypeScaleOff &&) = default;
    DTypeScaleOff &operator=(DTypeScaleOff const &) = default;
    DTypeScaleOff &operator=(DTypeScaleOff &&) = default;
    ~DTypeScaleOff() = default;
};

/**
 * For each 'interface' there are a pair of accessor classes
 *   Interface::Accessor
 *   Interface::AccessorRO
 *   .. which the types returned by Tensor(..indices...)
 *
 *  These have the following:
 *       typedef AccessorRO;                            - correponding RO type.
 *       typedef element_type;							- type of the stored element
 *       element_type .value() const;					- direct read
 *       .as_float() const;								- convert to float
 *       operator float() const;						- same
 *  (If not RO):
 *       .set_value( element_type &);                   - direct store
 *       .set_float( float )							- assign from float
 *       operator=( float )								- assign from float
 *       operator=( Accessor const & )				    - assign from same accessor
 *       operator=( AccessorRO const & )				- assign from R/O accessor
 *  The assignment operators may return either a float,
 *    or an AccessorRO by value
 *    or an Accessor const &  (which is *this)
 *    or an AccessorRO const & (only if it's *this by subclass).
 *
 *  Both have copy ctors, and AccessorRO(Accessor const &) works.
 *
 *  AccessorRO may or may not be a direct public base of Accessor
 *
 *  The 'GenericAccessor' and GenericAccessorRO have all of the above, except
 *  for element_type, .value(), and .set_value().
 */

/**
 * @class NullInterface
 *
 * @brief A NullInterface throws away data and returns zero
 */

class NullInterface final : public Interface {
    friend constexpr hnnx::ifc_method_table_t hnnx::construct_ifc_method_table();

  public:
    API_EXPORT constexpr NullInterface() : Interface(hnnx::dtype_info_v<DType::UNKNOWN>) {}

    static constexpr dtype_info get_dt_info() noexcept { return hnnx::dtype_info_v<DType::UNKNOWN>; }
    static qparms const *get_qparms() noexcept { return &Interface::null_parms; }
    static constexpr DType get_dtype() noexcept { return get_dt_info().dtype; }
    static constexpr unsigned element_size() noexcept { return get_dt_info().elbytes; }
    static constexpr bool is_quantized() noexcept { return get_dt_info().is_quant; }
    static uint32_t interface_hash() noexcept { return uint32_t(DType::UNKNOWN); }

  private:
    static void write_float(Interface const *, void * /*ptr*/, const float /*in*/) noexcept {}
    static float read_float(Interface const *, const void * /*ptr*/) noexcept { return 0.0f; }

    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    static constexpr intfc_methods get_method_table()
    {
        return {hnnx::dtype_info_v<DType::UNKNOWN>, read_float, write_float, get_null_qparms, nullptr, nullptr};
    }
    // LCOV_EXCL_STOP

  public:
    // hide the slower implementations in the base class...
    API_EXPORT float get_scale() const noexcept { return 1.0f; }
    API_EXPORT float get_scale_recip() const noexcept { return 1.0f; }
    API_EXPORT int32_t get_offset() const noexcept { return 0; }
    API_EXPORT int compare(const NullInterface & /* rhs */) const { return 0; }

    static hnnx::InterfaceRef get_refobj() noexcept
    {
        return hnnx::InterfaceRef(methods_for<DType::UNKNOWN>(), Interface::canonical_instance_for<NullInterface>());
    }
    // NullInterface has a null DTypeScaleOff
    static DTypeScaleOff get_dtype_scaleoff() noexcept { return DTypeScaleOff(); }

  private:
    // Accessor for NullInterface - empty class.
    struct nullval {
        operator float() const { return 0.0f; }
    };
    class AcsrRO {
      public:
        using element_type = nullval;
        using AccessorRO = AcsrRO;
        ALWAYSINLINE AcsrRO() {}
        ALWAYSINLINE AcsrRO(void const *, NullInterface const *) {}
        ALWAYSINLINE AcsrRO(AcsrRO const &) = default;
        ALWAYSINLINE AcsrRO(AcsrRO &&) = default;
        ALWAYSINLINE ~AcsrRO() = default;
        AcsrRO &operator=(AcsrRO const &) = delete;
        AcsrRO &operator=(AcsrRO &&) = delete;

        ALWAYSINLINE element_type value() const { return nullval{}; }
        ALWAYSINLINE float as_float() const { return 0.0f; }
        ALWAYSINLINE operator float() const { return 0.0f; }
    };
    class Acsr : public AcsrRO {
      public:
        using element_type = nullval;
        using AccessorRO = AcsrRO;
        ALWAYSINLINE Acsr(void *, const NullInterface *) {}
        ALWAYSINLINE Acsr(Acsr const &) = default;
        ALWAYSINLINE Acsr(Acsr &&) = default;
        ALWAYSINLINE ~Acsr() = default;
        ALWAYSINLINE void set_float(float /* v */) {}
        ALWAYSINLINE void set_value(element_type /* v */) {}
        ALWAYSINLINE float operator=(float /* v */) { return 0.0f; }
        ALWAYSINLINE float operator=(AcsrRO const & /* rhs */) { return 0.0f; }
        ALWAYSINLINE float operator=(Acsr const & /* rhs */) { return 0.0f; }
        ALWAYSINLINE float operator=(Acsr const &&) { return 0.0f; }
    };

  public:
    using Accessor = Acsr;
    using AccessorRO = AcsrRO;
};

// make_interface for NullInterface; easy, just have one
// and return a pointer to it.
template <> struct hnnx::make_interface<NullInterface> {
    API_EXPORT static Interface const *from_odef(Graph &, OutputDef const & /* odef */)
    {
        return Interface::canonical_instance_for<NullInterface>();
    }
    API_EXPORT static Interface const *from_deser(Deserz & /* dctx */, Interface const **)
    {
        return Interface::canonical_instance_for<NullInterface>();
    }
};

/**
 * @class PlainInterface
 *
 * @brief A tensor with Floats needs no conversion.
 * You could also use this for integral value tensors where the integral values are the true values;
 * they would get converted to floats.
 */
template <typename T> class PlainInterface final : public Interface {
    friend constexpr hnnx::ifc_method_table_t hnnx::construct_ifc_method_table();

  public:
    using element_type = T;
    static constexpr DType dtype = dtype_of_type<PlainInterface>();
    API_EXPORT explicit constexpr PlainInterface(const OutputDef & /* def */) : Interface(get_dt_info()) {}
    API_EXPORT constexpr PlainInterface() : Interface(get_dt_info()) {}
    API_EXPORT explicit constexpr PlainInterface(hnnx::Deserz &) : PlainInterface() {}
    API_EXPORT static constexpr T convert_from_float(const float &in)
    {
        return saturate_round<T>(in);
    } // except for T=float!
    API_EXPORT static constexpr float convert_to_float(const T &in) { return float(in); }

    static qparms const *get_qparms() noexcept { return &Interface::null_parms; }
    static constexpr dtype_info get_dt_info() noexcept { return hnnx::dtype_info_v<dtype>; }
    static constexpr DType get_dtype() noexcept { return dtype; }
    static constexpr unsigned element_size() noexcept { return get_dt_info().elbytes; }
    static constexpr bool is_quantized() noexcept { return get_dt_info().is_quant; }

  private:
    static void write_float(Interface const *self, void *ptr, const float in) noexcept; // inlined below
    static float read_float(Interface const *, const void *ptr) noexcept
    {
        auto p = static_cast<const T *>(ptr);
        return convert_to_float(*p);
    }
    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    static constexpr intfc_methods get_method_table()
    {
        return {hnnx::dtype_info_v<dtype>, read_float, write_float, get_null_qparms, nullptr, nullptr};
    }
    // LCOV_EXCL_STOP

  public:
    static uint32_t interface_hash() noexcept { return uint32_t(dtype); }
    static hnnx::InterfaceRef get_refobj() noexcept
    {
        return hnnx::InterfaceRef(methods_for<dtype>(), Interface::canonical_instance_for<PlainInterface>());
    }
    static DTypeScaleOff get_dtype_scaleoff() noexcept { return DTypeScaleOff(dtype); }

    API_EXPORT static int compare(const PlainInterface & /* rhs */) noexcept { return 0; }
    API_EXPORT static float get_scale() noexcept { return 1.0f; }
    API_EXPORT static float get_scale_recip() noexcept { return 1.0f; }
    API_EXPORT static int32_t get_offset() noexcept { return 0; }

  private:
    // Accessor for PlainInterface
    // Doesn't need a reference to interface, just a data pointer (or data, for AcsrRO)
    // We can't actually call it AccessorRO since it needs to contain a typedef AccessorRO.
    //
    class Acsr;
    class AcsrRO {
      protected:
        T val;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        AcsrRO(void const *data_in, PlainInterface const *) : val(*static_cast<T const *>(data_in)) {}
        AcsrRO(AcsrRO const &) = default;
        AcsrRO(AcsrRO &&) = default;
        AcsrRO &operator=(AcsrRO const &) = delete;
        AcsrRO &operator=(AcsrRO &&) = delete;
        ~AcsrRO() = default;
        ALWAYSINLINE AcsrRO(Acsr const &a) : val(a.value()) {}
        ALWAYSINLINE element_type value() const { return val; }
        ALWAYSINLINE float as_float() const { return convert_to_float(val); }
        ALWAYSINLINE operator float() const { return as_float(); }
    };
    class Acsr {
      protected:
        T *data;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        ALWAYSINLINE Acsr(void *data_in, PlainInterface const *) : data(static_cast<T *>(data_in)) {}
        ALWAYSINLINE Acsr(Acsr const &) = default;
        ALWAYSINLINE Acsr(Acsr &&) = default;
        ALWAYSINLINE ~Acsr() = default;
        ALWAYSINLINE element_type value() const { return *data; }
        ALWAYSINLINE float as_float() const { return convert_to_float(*data); }
        ALWAYSINLINE operator float() const { return as_float(); }

        ALWAYSINLINE void set_float(float v) { *data = convert_from_float(v); }
        ALWAYSINLINE void set_value(element_type v) { *data = v; }
        ALWAYSINLINE float operator=(float v)
        {
            set_float(v);
            return v;
        }
        // when copying from an Acsr of the same type we don't need to
        // convert to float and back.
        // @@we could also define operator= for other cases, e.g.
        //  int32 from int16, to do the operation without going to float.
        ALWAYSINLINE AcsrRO operator=(Acsr const &rhs)
        {
            if (this != &rhs) {
                T v = rhs.value();
                set_value(v);
            }
            return AcsrRO(*this);
        }
        // only needed for SA; forward to 'const &' operator=
        ALWAYSINLINE AcsrRO operator=(Acsr &&rhs) { return this->operator=(rhs); }

        ALWAYSINLINE AcsrRO operator=(AcsrRO const &rhs)
        {
            T v = rhs.value();
            set_value(v);
            return AcsrRO(*this);
        }
    };

  public:
    using Accessor = Acsr;
    using AccessorRO = AcsrRO;
};

//PlainInterface<float>::convert_from_float: no-op
template <> API_EXPORT inline constexpr float PlainInterface<float>::convert_from_float(const float &in)
{
    return in;
}

//PlainInterface<Float16>::convert_from_float: no-op for values in Float16 range, clamp to max otherwise.
template <> API_EXPORT inline constexpr Float16 PlainInterface<Float16>::convert_from_float(const float &in)
{
    return static_cast<Float16>(in);
}

//PlainInterface<BFloat16>::convert_from_float: no-op
template <> API_EXPORT inline constexpr BFloat16 PlainInterface<BFloat16>::convert_from_float(const float &in)
{
    return static_cast<BFloat16>(in);
}

// needs to be defined *after* convert_from_float is specialized
template <typename T> inline void PlainInterface<T>::write_float(Interface const *, void *ptr, const float in) noexcept
{
    auto p = static_cast<T *>(ptr);
    *p = convert_from_float(in);
}

// make_interface for PlainInterface<T>; easy, just have one
// and return a pointer to it.

template <typename T> struct hnnx::make_interface<PlainInterface<T>> {
    API_EXPORT static Interface const *from_odef(Graph &, OutputDef const & /* odef */)
    {
        return Interface::canonical_instance_for<PlainInterface<T>>();
    }
    API_EXPORT static Interface const *from_deser(Deserz & /* dctx */, Interface const **)
    {
        return Interface::canonical_instance_for<PlainInterface<T>>();
    }
};

extern template class PlainInterface<float>; // in tensor.cc
extern template class PlainInterface<NN_INT32_T>;
extern template class PlainInterface<NN_INT64_T>;

/**
 * @class ScaleOffsetInterface
 *
 * @brief A tensor could also have a scale+offset interface
 * This is good for quantization schemes where you want to quantize an arbitrary, possibly asymmetric range.
 * We compute and cache the reciprocal of the scale for conversion from float
 * A default constructor sets the offset to 0 and the scale to 1.0, which would be suitable for integers.
 * The reciprocal of scale should be computed so we don't have to divide.
 */

class SOIfcBase : public Interface { // base of ScaleOffsetInterface<T>
  protected:
    Interface::qparms qp; // offset, scale, scale_recip;
    template <typename X> friend Interface::qparms &hnnx::qparms_for_interface_patch(size_t);
    Interface::qparms &qparms_for_patch() { return qp; }
    // Can only be constructed by subclass.
    SOIfcBase(const int offset_, const float scale_, const dtype_info dt_)
        : Interface(dt_), qp({offset_, scale_, 1.0f / scale_})
    {
    }
    SOIfcBase(const OutputDef &def, const dtype_info dt_) : SOIfcBase(def.zero_offset, def.stepsize, dt_)
    {

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        if (def.stepsize == 0.0f) debuglog("Oops: zero stepsize");
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop // -Wfloat-equal
#endif
    }
    SOIfcBase(hnnx::Deserz &dctx, const dtype_info dt_) : Interface(dt_)
    {
        qp.offset = static_cast<int>(dctx.deserialize_uint32());
        qp.scale = dctx.deserialize_float();
        qp.scale_recip = 1.0f / qp.scale;
    }

    // I assume there is good reason for these float comparisons
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    // these two are protected here (they only make sense in comparing same type)
    // but are exposed in subclass via wrappers
    API_EXPORT int compare(const SOIfcBase &rhs) const noexcept
    {

        if (qp.offset != rhs.qp.offset) return (qp.offset < rhs.qp.offset) ? -1 : 1;
        if (qp.scale != rhs.qp.scale) return (qp.scale < rhs.qp.scale) ? -1 : 1;
        return 0;
    }
    API_EXPORT bool compare_eq(const SOIfcBase &rhs) const noexcept
    {
        return qp.offset == rhs.qp.offset && qp.scale == rhs.qp.scale;
    }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop // -Wfloat-equal
#endif
  public:
    API_EXPORT float get_scale() const noexcept { return qp.scale; }
    API_EXPORT float get_scale_recip() const noexcept { return qp.scale_recip; }
    API_EXPORT int32_t get_offset() const noexcept { return qp.offset; }
    API_EXPORT qparms const *get_qparms() const noexcept { return &qp; }

  protected:
    static Interface::qparms const *get_qparms_meth(Interface const *const self) noexcept
    {
        return &static_cast<SOIfcBase const &>(*self).qp;
    }
    static int ifc_compare(Interface const *const lhs, Interface const *const rhs) noexcept
    {
        auto const &rhs_ref = *static_cast<SOIfcBase const *>(rhs);
        return static_cast<SOIfcBase const *>(lhs)->compare(rhs_ref);
    }
    static uint32_t ifc_hash(Interface const *const self) noexcept
    {
        Interface::qparms const *qpp = get_qparms_meth(self);
        // NOTE; it's important that if two ScaleOffsetInterface<T> objects for two *different*
        // T have the same scale and offset, they must have different hash values. So 'dtype'.
        // is rolled into the hash. Hash collisions are OK if either scale or offset is different.
        return unsigned(qpp->offset) * 0x10661 ^ (image_convert<unsigned, float>(qpp->scale) << 1);
    }
};

template <typename T> class ScaleOffsetInterface final : public SOIfcBase {
    friend constexpr hnnx::ifc_method_table_t hnnx::construct_ifc_method_table();

  public:
    API_EXPORT ScaleOffsetInterface(const int offs_, const float scale_) : SOIfcBase(offs_, scale_, get_dt_info()) {}
    API_EXPORT explicit ScaleOffsetInterface(const OutputDef &def) : SOIfcBase(def, get_dt_info()) {}
    API_EXPORT ScaleOffsetInterface() : SOIfcBase(0, 1.0f, get_dt_info()) {}
    API_EXPORT explicit ScaleOffsetInterface(hnnx::Deserz &dctx) : SOIfcBase(dctx, get_dt_info()) {}

    using element_type = T;
    static constexpr DType dtype = dtype_of_type<ScaleOffsetInterface>();
    template <typename TX> API_EXPORT static constexpr T saturate(TX in) { return saturate_cast<T>(in); }
    API_EXPORT constexpr T convert_from_float(float in) const
    {
        return saturate_round<T>(qp.offset + in * qp.scale_recip);
    }
    API_EXPORT constexpr float convert_to_float(T in) const
    {
        if constexpr (sizeof(T) <= 2)
            return (float(int(in) - qp.offset)) * qp.scale;
        else
            return (float(in) - qp.offset) * qp.scale;
    }
    static constexpr dtype_info get_dt_info() noexcept { return hnnx::dtype_info_v<dtype>; }
    static constexpr DType get_dtype() noexcept { return dtype; }
    static constexpr unsigned element_size() noexcept { return get_dt_info().elbytes; }
    static constexpr bool is_quantized() noexcept { return get_dt_info().is_quant; }

  private:
    static void write_float(Interface const *const self, void *ptr, const float in) noexcept
    {
        assert(ptr != nullptr);
        auto p = static_cast<T *>(ptr);
        *p = static_cast<ScaleOffsetInterface<T> const *>(self)->convert_from_float(in);
    }
    static float read_float(Interface const *const self, const void *ptr) noexcept
    {
        assert(ptr != nullptr);
        auto p = static_cast<const T *>(ptr);
        return static_cast<ScaleOffsetInterface<T> const *>(self)->convert_to_float(*p);
    }
    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    static constexpr intfc_methods get_method_table()
    {
        return {hnnx::dtype_info_v<dtype>, read_float, write_float, get_qparms_meth, ifc_hash, ifc_compare};
    }
    // LCOV_EXCL_STOP

  public:
    uint32_t interface_hash() const noexcept { return ifc_hash(this) ^ uint32_t(dtype); }

    hnnx::InterfaceRef get_refobj() const noexcept { return hnnx::InterfaceRef(methods_for<dtype>(), this); }
    DTypeScaleOff get_dtype_scaleoff() const noexcept { return DTypeScaleOff(dtype, *get_qparms_meth(this)); }

    API_EXPORT int compare(const ScaleOffsetInterface &rhs) const noexcept { return SOIfcBase::compare(rhs); }
    API_EXPORT bool compare_eq(const ScaleOffsetInterface &rhs) const noexcept { return SOIfcBase::compare_eq(rhs); }

  private:
    // Accessor for ScaleOffsetInterface
    class Acsr;
    class AcsrRO {
      protected:
        T val;
        const ScaleOffsetInterface<T> &interface;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        ALWAYSINLINE AcsrRO(void const *data_in, const ScaleOffsetInterface *interface_in)
            : val(*static_cast<T const *>(data_in)), interface(*interface_in)
        {
        }

        AcsrRO(AcsrRO const &) = default;
        AcsrRO(AcsrRO &&) = default;
        ~AcsrRO() = default;
        AcsrRO &operator=(AcsrRO const &) = delete;
        AcsrRO &operator=(AcsrRO &&) = delete;
        ALWAYSINLINE element_type value() const { return val; }
        ALWAYSINLINE float as_float() const { return interface.convert_to_float(val); }
        ALWAYSINLINE operator float() const { return as_float(); }
        ALWAYSINLINE AcsrRO(Acsr const &a) : val(a.value()), interface(a.interface) {}
    };
    class Acsr {
        friend class AcsrRO;

      protected:
        T *data;
        const ScaleOffsetInterface<T> &interface;

      public:
        using element_type = T;
        using AccessorRO = AcsrRO;
        ALWAYSINLINE Acsr(void *data_in, const ScaleOffsetInterface *interface_in)
            : data(static_cast<T *>(data_in)), interface(*interface_in)
        {
        }
        ALWAYSINLINE Acsr(Acsr const &) = default;
        ALWAYSINLINE Acsr(Acsr &&) = default;
        ALWAYSINLINE ~Acsr() = default;
        ALWAYSINLINE element_type value() const { return *data; }
        ALWAYSINLINE float as_float() const { return interface.convert_to_float(*data); }
        ALWAYSINLINE operator float() const { return as_float(); }
        ALWAYSINLINE void set_float(float v) { *data = interface.convert_from_float(v); }
        ALWAYSINLINE void set_value(element_type v) { *data = v; }
        ALWAYSINLINE float operator=(float v)
        {
            set_float(v);
            return v;
        }
        ALWAYSINLINE float operator=(Acsr const &rhs)
        {
            if (this != &rhs) {
                float const v = rhs.as_float();
                set_float(v);
                return v;
            }
            return this->as_float();
        }
        // only needed for SA; forward to 'const &' operator=
        ALWAYSINLINE float operator=(Acsr &&rhs) { return this->operator=(rhs); }

        ALWAYSINLINE float operator=(AcsrRO const &rhs)
        {
            float const v = rhs.as_float();
            set_float(v);
            return v;
        }
    };

  public:
    using Accessor = Acsr;
    using AccessorRO = AcsrRO;
};

// make_interface for ScaleOffsetInterface.
template <typename T> struct hnnx::make_interface<ScaleOffsetInterface<T>> {
    // can only declare these here, since we can't see into Graph at this point.
    // Code is in tensor.cc
    API_EXPORT static ScaleOffsetInterface<T> const *from_exemplar(Graph &, ScaleOffsetInterface<T> const &exemplar);
    API_EXPORT static Interface const *from_odef(Graph &g, OutputDef const &odef)
    {
        // make an exemplar...
        ScaleOffsetInterface<T> const exemplar(odef);
        return from_exemplar(g, exemplar);
    }
    API_EXPORT static Interface const *from_deser(Deserz &dctx, Interface const **const ptrloc)
    {
        // deserialize the id; p is a pointer to where it is in index,
        const auto [objp, indexp] = dctx.deserialize_shared_obj<Interface>(ptrloc);
        // if indexp is null, it's a ref to previous obj; 'objp' is the pointer and we're done.
        if (indexp == nullptr) return objp;
        // otherwise, make a new one and store its address at indexp.
        Interface const *const new_p = dctx.dcrate()->emplace0<ScaleOffsetInterface<T>>(dctx);
        *indexp = new_p; // for next time it's used
        return new_p;
    }

  protected:
    // put in crate without checking for dups.
    API_EXPORT static ScaleOffsetInterface<T> const *to_crate(Graph &, ScaleOffsetInterface<T> const &exemplar);
};

extern template class ScaleOffsetInterface<uint8_t>; // in tensor.cc
extern template class ScaleOffsetInterface<uint16_t>;

POP_VISIBILITY()

#endif // HEXNN_INTERFACE_H
