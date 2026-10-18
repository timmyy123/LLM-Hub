// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef TYPICAL_OP_H
#define TYPICAL_OP_H 1

#include "ctor_hook.h"
#include "graph_status.h"
#include "op.h"
#include "op_def.h"
#include "op_utils.h"
#include "op_io_ptrs.h"
#include "tensor.h"
#include "op_info.h"
#include "bake_defs.h"
#include "weak_linkage.h"
#include "macros_attribute.h"
#include "build_options_pub.h"

#include <array>
#include <cassert>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

/*
 * What do we need to do at op creation time?
 *
 * When we create an op, we don't want to try to find the input tensors right away,
 * because the op might not have been created, or might get modified as the graph
 * optimization process.  Instead we can record the op ID and which output of the op
 * should be used.
 *
 * Similarly, we probably want to keep the definitions of the output tensors
 * rather than create them right away, since we may modify the op later on.
 * Although eager creation of the output tensors does solve some other issues.
 *
 * After we've optimized the graph, we can go through and allocate the output tensors
 * as needed, and link the inputs to the appropriate places.
 *
 */

/*
 * Input and output types
 *
 * As we saw in tensor.h, we would like to use the type of the tensors to help
 * enforce what kind of tensors can be attached to what kinds of ops; ops should
 * be able to mandate dimentionality or internal type or layout (or all of the above)
 * and have the system enforce that only those kinds of tensors will be attached.
 *
 * We could create a separate class for each op, inheriting from a base class for the
 * required interfaces, and adding custom code or values for that ops' inputs and outputs.
 * But this is quite cumbersome and error-prone, and if we need to add a new interface
 * to an op (such as the dependence graph informtation), we would potentially need to update
 * it in several places.
 *
 * While this is straightforward, it requires too much duplicated code and if something
 * needs to be the same in two places, it won't be.
 *
 * Furthermore, the op writers are not C++ people and the idea of creating a class is
 * foreign, intimindating, and not making the best use of time.
 *
 * The execution function for every op is different, and is the critical, fundamental
 * part of the op.  We should strive to make this execution function easy to write,
 * and use the execution function to help us know the input and output types.
 *
 */

/*
 * Execution Function and Data Types
 *
 * Our execution function needs to have:
 * * Specification for inputs, tying them to variables to utilize
 * * Specification for outputs, tying them to variables to utilize
 * * The instructions for computation
 * * A method for indicating failure
 *
 *
 * This is well-known in computation: these things are what almost every function needs to do.
 * Indeed, what does a prototype look like for a simple instruction?  It will need an input,
 * it will need an output (which is pre-allocated, so it's not a return value, but a pointer/reference
 * to something that gets filled in...), and let's give it a hook to indicate failure.  Since the results
 * are not returned, we can have a simple integer return value to indicate failure.
 *
 * int nop_impl(GenericTensor &out, const GenericTensor &in);
 *
 * This has all the information we need:
 * * The number and types of the outputs: 1 output tensor of type "GenericTensor"
 * * The number and types of the inputs:  1  input tensor of type "const GenericTensor" (inputs are const)
 * * A return value to indicate failure
 * * The location of the execution function is at the symbole nop_impl.
 *
 *
 *
 * This function prototype is within the abilities of even the most C++ - averse assembly programmer.
 * All we need to do is create all the C++ information from this function definition.
 *
 * Fortunately, with the power of templates, we can do this in C++.
 *
 * Here's the basic strategy:
 * * The function is a template parameter [template <auto F,...]
 * * The type of the function is obtained [decltype(F)]
 * * We find the arguments of the function
 * * The const arg types are filtered into inputs
 * * The non-const arg types are filtered into outputs
 * * We can use tuples to hold lists of types
 * * Number of inputs and outputs can be static constexpr values, not taking up space in the op
 * * Fixed-size arrays of input and output pointers are stored in the op,
 *   rather than extra allocations being required (lots of small allocations in
 *   current Hexagon NN are inefficient)
 *
 * For execution, we can take the tensor pointers, cast them to the appropriate type and deference,
 * keeping them in appropriately-typed tuples, and then applying the tuple to the templated execution function F.
 *
 * On x86, this yields the generation of an execute() function that loads up the pointers out of the arrays and
 * calls the execution function, directly (not through a pointer) and immediately.  This is the behavior we want.
 *
 * In a nutshell:
 * * We can use the power of C++ to generate custom classes for most ops given an execution function
 * * It requires some substantial template trickery, but it seems to work
 * * The results appear to be fast at runtime
 * * The cost over writing an independent class::execute() function is negligable (direct branch)
 *
 */

/*
  * It doesn't work everywhere
  * This scheme doesn't work everywhere.  In particular, I think some special op categories will
  * need other facilities (probably just doing all the class development by hand):
  *
  * * Variardic ops.  The number and type of elements may not be known until runtime.
  * * Ops that may need to do special things, like Const or Variable or Assign
  *
  * But I think it will work for the vast majority of cases.
  */

/*
 * To ensure that ops are valid for the architecture, detect how the op is compiled,
 * and ensure that we can match that minimum architecture / features / etc
 */

/*
 * FIXME: do we reduce some code size if TypicalOp inherits everything except the execute() function
 * from an op that fills out everything else based on the input/output types from the execution
 * function's parameters?  I think we're getting duplicated constructors / valid_construction /
 * get_output / etc just with different functions.
 *
 * It seems like it should be simple to split out the non-function-specific stuff.
 */

/*
 * FIXME: tensor serialization/deserialization should update its own ID information, rather than put it in op...
 */

#ifdef __HVX__
#define USES_HVX __HVX__
#else
#define USES_HVX 0
#endif

#ifndef MIN_ARCH
#ifdef __HEXAGON_ARCH__
#define MIN_ARCH __HEXAGON_ARCH__
#else
#define MIN_ARCH 0
#endif
#endif

PUSH_VISIBILITY(hidden)

namespace hnnx {

//
//  Op
//   TypicalOpUtil
//      TypicalOpIoBase<unsigned NOUT, unsigned NIN>
//         TypicalOpIo<typename FType>
//            TypicalOp<auto [FType*] F>

/// @brief TypicalOpUtil contains non-templated methods
/// to help implement the templated methods of TypicalOpIoBase and TypicalOpIO
PUSH_VISIBILITY(default)
class TypicalOpUtil : public Op {
  protected:
    explicit TypicalOpUtil(OpIoPtrs const &ioptrs) : Op(ioptrs.graph(), ioptrs.get_id()) {}
    explicit TypicalOpUtil(Deserz &dctx) : Op(dctx) {}
    TypicalOpUtil(const TypicalOpUtil &) = delete;
    TypicalOpUtil &operator=(const TypicalOpUtil &) = delete;
    TypicalOpUtil(TypicalOpUtil &&) = delete;
    TypicalOpUtil &operator=(TypicalOpUtil &&) = delete;
    virtual ~TypicalOpUtil() override;

#ifndef PREPARE_DISABLED
    virtual Flags_word get_flag_word() const override;
#endif
    virtual const char *get_docs() const override { return hnnx::docs_for<TypicalOpUtil>(); }

    // this is called from TypicalOpIoBase when deserializing
    void do_deserialize(Deserz &dctx, size_t n_in, Tensor const **inputs, size_t n_out, uptr_Tensor *outputs);

    GraphStatus do_allocate(Graph &graph_in, size_t n_out, uptr_Tensor *outputs);

    // used in TypicalOpIO::prepare
    GraphStatus assign_input_pointers(OpIoPtrs const &op_io_ptrs, size_t n_inputs, Tensor const **inputs);
    GraphStatus set_input_pointer(size_t which, Tensor const *input);

    GraphStatus output_create(OpIoPtrs const &op_io_ptrs, size_t num_outputs, uptr_Tensor *outputs,
                              tensor_generate_fp const *out_gen_functions);
    GraphStatus output_scratch_create(Graph &gr, size_t num_scratch_outputs, uptr_Tensor *scratch_outputs,
                                      tensor_generate_fp const *out_gen_functions, dt_rank_pair const *dt_rank_vals);
    GraphStatus output_allocate(OpIoPtrs const &op_io_ptrs, size_t num_outputs, uptr_Tensor *outputs,
                                tensor_generate_fp const *out_gen_functions);
    GraphStatus output_allocate_with_scratch(OpIoPtrs const &op_io_ptrs, size_t num_outputs, size_t num_scratch_outputs,
                                             uptr_Tensor *outputs, tensor_generate_fp const *out_gen_functions,
                                             dt_rank_pair const *dt_rank_vals);
};

template <unsigned N_OUT, unsigned N_IN> struct TypIoIo {
    // Pointers to the inputs.  These are kept as generic Tensor pointers,
    // we check @ prepare and (static) downcast during execute
    std::array<const Tensor *, N_IN> inputs_arr;
    // unique_ptrs to the outputs.  We own our outputs.
    // We (static) downcast during execute
    std::array<uptr_Tensor, N_OUT> outputs_arr;

    std::array<const Tensor *, N_IN> &inputs() { return inputs_arr; }
    std::array<const Tensor *, N_IN> const &inputs() const { return inputs_arr; }
    std::array<uptr_Tensor, N_OUT> &outputs() { return outputs_arr; }
    std::array<uptr_Tensor, N_OUT> const &outputs() const { return outputs_arr; }
};

extern std::array<uptr_Tensor, 0> typical_op_0_outputs;
extern std::array<const Tensor *, 0> typical_op_0_inputs;

// TypIoIo with 0 outputs.
// This is done because std::array<uptr_Tensor, 0> occupies the same space as if its size is 1.
// in some implementations it only occupies 1 byte. We can make it go away entirely this way.
template <unsigned N_IN> struct TypIoIo<0, N_IN> {
    std::array<const Tensor *, N_IN> inputs_arr;

    std::array<const Tensor *, N_IN> &inputs() { return inputs_arr; }
    std::array<const Tensor *, N_IN> const &inputs() const { return inputs_arr; }
    std::array<uptr_Tensor, 0> &outputs() { return typical_op_0_outputs; }
    std::array<uptr_Tensor, 0> const &outputs() const { return typical_op_0_outputs; }
};
template <unsigned N_OUT> struct TypIoIo<N_OUT, 0> {
    std::array<uptr_Tensor, N_OUT> outputs_arr;

    std::array<const Tensor *, 0> &inputs() { return typical_op_0_inputs; }
    std::array<const Tensor *, 0> const &inputs() const { return typical_op_0_inputs; }
    std::array<uptr_Tensor, N_OUT> &outputs() { return outputs_arr; }
    std::array<uptr_Tensor, N_OUT> const &outputs() const { return outputs_arr; }
};
template <> struct TypIoIo<0, 0> {
    std::array<const Tensor *, 0> &inputs() { return typical_op_0_inputs; }
    std::array<const Tensor *, 0> const &inputs() const { return typical_op_0_inputs; }
    std::array<uptr_Tensor, 0> &outputs() { return typical_op_0_outputs; }
    std::array<uptr_Tensor, 0> const &outputs() const { return typical_op_0_outputs; }
};

template <unsigned N_OUT, unsigned N_IN> class TypicalOpIoBase : public TypicalOpUtil {
    static constexpr size_t n_inputs = N_IN;
    static constexpr size_t n_outputs = N_OUT;

  protected:
    explicit TypicalOpIoBase(OpIoPtrs const &ioptrs) : TypicalOpUtil(ioptrs) {}
    explicit TypicalOpIoBase(Deserz &dctx) : TypicalOpUtil(dctx)
    {
        // NOTE: must use inputs.data() here, not &inputs[0]; when n_inputs==0,
        // evaluating &inputs[0] generates an abort.
        do_deserialize(dctx, n_inputs, io.inputs().data(), n_outputs, io.outputs().data());
    }

    virtual bool swap_output(size_t which, uptr_Tensor &value) override
    {
        if (which < n_outputs) {
            uptr_Tensor &out = io.outputs()[which];
            if (!value || !out) {
                std::swap(out, value);
                return true;
            }
        }
        return false;
    }
    // on hexagon builds, this checks that the size of TypicalOpIoBase<N_OUT,N_IN> is as expected.
    // in TypicalOp::execute method, we static-assert this, and check that the subclass is the same size as this base.
    inline static constexpr bool check_szal_base()
    {
        constexpr auto claimed_size = bake::typical_op_tgt_size_align(n_inputs, n_outputs);
        return bake::check_size_align<TypicalOpIoBase>::template check<std::get<0>(claimed_size),
                                                                       std::get<1>(claimed_size)>();
    }

  public:
    TypIoIo<n_outputs, n_inputs> io; // pointers to inputs and outputs

  protected:
    virtual const Tensor *get_input_output(size_t which, bool is_input) const override
    {
        if (is_input) {
            assert(which < n_inputs);
            return io.inputs()[which];
        } else {
            assert(which < n_outputs);
            return io.outputs()[which].get();
        }
    }

  public:
    virtual bool set_input(size_t which, const Tensor *tensor) override
    {
        assert(which < n_inputs);
        std::swap(io.inputs().at(which), tensor);
        return true;
    }

    virtual std::pair<size_t, size_t> num_inputs_outputs() const override { return {n_inputs, n_outputs}; }
    virtual GraphStatus allocate(Graph &graph_in) override
    {
        if constexpr (n_outputs > 0) {
            return do_allocate(graph_in, n_outputs, io.outputs().data());
        } else {
            return GraphStatus::Success;
        }
    }

    // Right now this seems reasonable: a function to check whether this candidate op fits.
    // It might not for a few reasons:
    // * It requires an architecture or feature not available (Hexagon architecture version or HVX present, etc)
    // ** But, this we could exclude from even entering the ops that are registered, right?
    // * The inputs are invalid types
    // ** For example, if we mandate a tensor with true elements of float type, we can't accept one with quint8 types.
    // ** This is important for us to check, so that we can static_cast at execute() time.
    // ** This check can be done by dynamic_cast and check for nullptr I think.
    /* Is this duplicated with valid_construction? */
    virtual bool is_valid() const noexcept override
    {
        /* Find input tensors */
        /* Ensure input tensors are compatible with our types */
        /* Check Hexagon version */
        return true;
    }

    virtual void enumerate_blocks(MemBlockEnumerator &en, bool is_input) const override
    {
        // template method of Op:
        enumerate_op_blocks(en, io.inputs(), io.outputs(), is_input);
    }

    virtual void serialize(SerOpsInterface &sctx) const override { sctx.op_typical(this, io.inputs(), io.outputs()); }
};

// these cases are fully specialized in typical_op.cc
extern template class TypicalOpIoBase<0, 1>;
extern template class TypicalOpIoBase<1, 0>;
extern template class TypicalOpIoBase<1, 1>;
extern template class TypicalOpIoBase<1, 2>;
extern template class TypicalOpIoBase<1, 3>;
extern template class TypicalOpIoBase<1, 4>;
extern template class TypicalOpIoBase<1, 5>;
extern template class TypicalOpIoBase<1, 6>;
extern template class TypicalOpIoBase<2, 1>;
POP_VISIBILITY()

/*
 * Having the OpIO split out allows ops with the same signatures to share code.
 *
 * I'd like to refine this further.
 *
 * In particular, it would be nice to:
 * * Share as much code as possible
 * * Enable common infrastructure for variadic op defs
 */

template <typename Ftype>
class TypicalOpIO : public TypicalOpIoBase<ArgsTuples<Ftype>::n_outputs, ArgsTuples<Ftype>::n_inputs> {
  protected:
    using arg_tuples = ArgsTuples<Ftype>;
    static_assert(std::tuple_size_v<typename arg_tuples::var_input_tuple> == 0 &&
                          std::tuple_size_v<typename arg_tuples::var_output_tuple> == 0,
                  "VECTOR args not allowed in TypicalOp");

  public:
    // The collection of input types, as pointers
    using input_tuple_type = typename ArgsTuples<Ftype>::input_ptr_tuple;
    // The collection of output types, as pointers
    using output_tuple_type = typename ArgsTuples<Ftype>::output_ptr_tuple;
    // the outputs as real types
    using output_tuple_defs = typename ArgsTuples<Ftype>::output_tuple;

    // The collection of output types, as std::unique_ptr s
    using output_uniqueptrs_tuple_type = typename ArgsTuples<Ftype>::output_uniqueptrs_tuple;
    // Actual output types
    using true_output_tuple_type = typename ArgsTuples<Ftype>::output_tuple;

    static constexpr size_t n_inputs = ArgsTuples<Ftype>::n_inputs;
    static constexpr size_t n_outputs = ArgsTuples<Ftype>::n_outputs;
    static constexpr size_t n_scratch_outputs = ArgsTuples<Ftype>::n_scratch_outputs;
    static constexpr size_t n_nonscratch_outputs = n_outputs - n_scratch_outputs;

    //TypicalOpIO(Graph &graph_in, OpId my_id_in, const OpDef *op_def_in)
    explicit TypicalOpIO(OpIoPtrs const &ioptrs) : TypicalOpIoBase<n_outputs, n_inputs>(ioptrs) {} // see note below...
    explicit TypicalOpIO(Deserz &dctx) : TypicalOpIoBase<n_outputs, n_inputs>(dctx) {}

    void assign_input_output_pointers(OpIoPtrs const &ioptrs)
    {
        this->assign_input_pointers(ioptrs, n_inputs, this->io.inputs().data());
        auto const *gen_arr_p = output_generator_array();
        // Different behaviour for cloning when 'scratch' outputs are present:
        // when cloning, the full output count is supplied to output_create,
        // and output_scratch_create is called as usual.
        unsigned n_outputs_to_create = n_nonscratch_outputs;
        if (n_scratch_outputs > 0 && ioptrs.is_clone_mode()) {
            n_outputs_to_create += n_scratch_outputs;
        }
        this->output_create(ioptrs, n_outputs_to_create, this->io.outputs().data(), gen_arr_p);
        if constexpr (n_scratch_outputs > 0) {
            this->output_scratch_create(ioptrs.graph(), n_scratch_outputs,
                                        this->io.outputs().data() + n_nonscratch_outputs,
                                        gen_arr_p + n_nonscratch_outputs, scratch_output_dt_rank_array());
        }
    }
    static bool valid_construction(OpIoPtrs const &op_io_ptrs, const OpId /*my_id_in*/)
    {
        Graph &graph_in = op_io_ptrs.graph();
        size_t const n_inputs_in = op_io_ptrs.n_inputs();
        size_t const n_outputs_in = op_io_ptrs.n_outputs();
        unsigned n_outs_expected = n_nonscratch_outputs;
        if (n_scratch_outputs > 0 && op_io_ptrs.is_clone_mode()) {
            n_outs_expected += n_scratch_outputs;
        }
        //debuglog("n_inputs_in=%zd n_outputs_in=%zd",n_inputs_in,n_outputs_in);
        //debuglog("expected n_inputs=%zd n_outputs=%zd",n_inputs,n_outputs);
        if (n_outs_expected != n_outputs_in) return false;
        if (n_inputs != n_inputs_in) return false;
        //debuglog("numbers OK");
        if (!are_input_tensors_compatible<n_inputs, input_tuple_type>(graph_in, op_io_ptrs.in_tensors.data()))
            return false;
        if (build_options_pub::DebugRegistry) debuglog("inputs OK");
        if (!op_io_ptrs.is_clone_mode()) {
            if (!op_io_ptrs.out_defs.empty()) {
                if (!are_output_defs_valid<n_nonscratch_outputs, output_tuple_defs>(op_io_ptrs.out_defs.data(),
                                                                                    graph_in)) {
                    return false;
                }
            }
        }
        //debuglog("outputs OK");
        return true;
    }

    // static method: its result is a constant ptr and depends only on the class template parameters.
    static tensor_generate_fp const *output_generator_array() { return tensor_gen_array_ptr<true_output_tuple_type>(); }
    // Return a pointer to a const array of [n_scratch_outputs] of dt_rank_pair, each of which is the dt,rank for
    // one of the scratch output tensors; it will be nullptr if there are none.
    static inline dt_rank_pair const *scratch_output_dt_rank_array()
    {
        if constexpr (n_scratch_outputs == 0) {
            return nullptr;
        } else {
            // this discards the first 'n_nonscratch_outputs' values in true_output_tuple_type, and builds
            // a table from the rest.
            return tensor_dt_rank_array_for_scratch<n_nonscratch_outputs, true_output_tuple_type>::table_p();
        }
    }
    // We could allocate outputs right when we create the op
    // However it might make sense to defer allocation, since we will probably create and destroy
    // ops quite a lot during graph optimization.  For now we will just do it during construction.

    virtual GraphStatus prepare(OpIoPtrs const &op_io_ptrs, bool tcm_available) override
    {
        GraphStatus result = GraphStatus::Success;
        if constexpr (build_options_pub::WITH_PREPARE) {
            bool needs_tcm = has_memclass<MemoryClass::TCM, output_tuple_defs>::value;
            if (needs_tcm && !tcm_available) return GraphStatus::ErrorNoTCM;

            /* Assign input pointers */
            /* this also calls hooks, if present, so must call even when n_inputs==0 */
            result = this->assign_input_pointers(op_io_ptrs, n_inputs, this->io.inputs().data());

            /* Maybe allocate outputs? */
            /* (also calls hooks) */
            if (result == GraphStatus::Success) {
                if constexpr (n_scratch_outputs == 0) {
                    result = this->output_allocate(op_io_ptrs, n_outputs, this->io.outputs().data(),
                                                   output_generator_array());
                } else {
                    result = this->output_allocate_with_scratch(op_io_ptrs, n_outputs, n_scratch_outputs,
                                                                this->io.outputs().data(), output_generator_array(),
                                                                scratch_output_dt_rank_array());
                }
            }
            // this->dependencies_left = this->dependencies;
        }
        return result;
    }

    // there are fewer combinations of true_output_tuple_type than there are
    // TypicalOpIO, so it's better to return a function ptr here than to make one.
    //
    static constexpr Op::tensor_deserializer_register_func get_tensor_deserializer_register_func()
    {
        return hnnx::deserialize_tensor_tuple<true_output_tuple_type, false>::f_ptr();
    }

  protected:
    static constexpr bool has_slice_parm = arg_tuples::has_slice_spec;
    static constexpr bool has_graph_parm = arg_tuples::has_graph;

    static constexpr size_t parm_n_inout = n_outputs + n_inputs;
    static constexpr size_t parm_n_total = n_outputs + n_inputs + (has_slice_parm ? 1 : 0) + (has_graph_parm ? 1 : 0);

    // generate parameter I (in range 0..n_in+n_out-1) for calling the func within execute.
    // Return type is auto &, will always return a reference.
    // This is only used for outputs & inputs & scratch outputs; not for 'extras' (slice_spec, Graph const &)
    template <size_t I> inline auto &get_exec_parm() const noexcept
    {
        static_assert(I < parm_n_inout);
        if constexpr (I < n_nonscratch_outputs || I >= n_nonscratch_outputs + n_inputs) { // output (incl. scratch)
            static constexpr size_t Iout = I - ((I >= n_nonscratch_outputs) ? n_inputs : 0);
            using output_ptr_t = std::tuple_element_t<Iout, output_tuple_type>;
            // extract output[Iout], downcast to output_ptr_t, return ref
            return *static_cast<output_ptr_t>(this->io.outputs()[Iout].get());
        } else {
            using input_ptr_t = std::tuple_element_t<I - n_nonscratch_outputs, input_tuple_type>;
            // extract input[I - n_nonscratch_outputs], downcast to input_ptr_t, return ref
            return *static_cast<input_ptr_t>(this->io.inputs()[I - n_nonscratch_outputs]);
        }
    }
    template <size_t... I>
    inline GraphStatus call_with_parms(Ftype f, Graph *gp, op_slice_spec ss, std::index_sequence<I...>) const noexcept
    {
        GraphStatus result = GraphStatus::ErrorFatal;
        if constexpr (!has_graph_parm) { // decide which 'extra' parms we need...
            if constexpr (!has_slice_parm) {
                result = GraphStatus(f(get_exec_parm<I>()...));
            } else {
                result = GraphStatus(f(get_exec_parm<I>()..., ss));
            }
        } else {
            if constexpr (!has_slice_parm) {
                result = GraphStatus(f(get_exec_parm<I>()..., *gp));
            } else {
                result = GraphStatus(f(get_exec_parm<I>()..., ss, *gp));
            }
        }
        return result;
    }
};

template <auto F>
class TypicalOp : public TypicalOpIO<std::remove_pointer_t<decltype(F)>> //, public Cost<TypicalOp<F>>
{
    using Ftype = std::remove_pointer_t<decltype(F)>;

  public:
    using ThisType = TypicalOp<F>;
    using ThisIoBase = TypicalOpIoBase<TypicalOpIO<Ftype>::n_outputs, TypicalOpIO<Ftype>::n_inputs>;

    using typename TypicalOpIO<Ftype>::output_tuple_type;
    // Values that are instance-invariant and knowable at compile time
    static constexpr size_t min_arch = MIN_ARCH; // FIXME: move somewhere else?
    static constexpr size_t needs_hvx = USES_HVX;

    // Definitions for inputs.  These aren't needed after we prepare the graph, so perhaps they could be moved out
    //std::array<InputDef,n_inputs> input_defs;
    // Definitions for outputs.  These aren't needed after we prepare the graph, so perhaps they could be moved out
    //std::array<OutputDef,n_outputs> output_defs;

    // construtor.
    explicit TypicalOp(Deserz &dctx) : TypicalOpIO<Ftype>(dctx) {}

    virtual GraphStatus prepare(OpIoPtrs const &op_io_ptrs, bool tcm_available) override
    {
        GraphStatus ret = GraphStatus::Success;
        if constexpr (build_options_pub::WITH_PREPARE) {
            Graph &graph_in = op_io_ptrs.graph();
            ctor_ophook<TypicalOp>(op_io_ptrs);
            ret = TypicalOpIO<Ftype>::prepare(op_io_ptrs, tcm_available);
            if (ret == GraphStatus::Success) ctor_hook(graph_in, *this);
        }
        return ret;
    }

    // Generator function.  Create the op if possible, otherwise return null.
    static uptr_Op create(OpIoPtrs const &ioptrs, const OpId my_id_in)
    {
        // Make sure we can create it.
        if (!ThisType::valid_construction(ioptrs, my_id_in)) return uptr_Op{};
        ThisType *op = new ThisType{ioptrs};
        // Hook in the inputs/outputs early so we can use cost()
        op->assign_input_output_pointers(ioptrs);
        return uptr_Op{op};
    }

    // Typical execute function.
    // Get the inputs and outputs and apply them to the function
    // This should turn into efficient code that pulls out the inputs to their correct place as arguments
    // then sibling calls the templated function directly.
    // Performance is very important for execute()
    virtual GraphStatus execute(Graph *const g, op_slice_spec const slice_spec_in) const noexcept override
    {
        static_assert(ThisIoBase::check_szal_base());
        static_assert(sizeof(TypicalOp) == sizeof(ThisIoBase));
        op_slice_spec slice_spec{};
        if constexpr (TypicalOpIO<Ftype>::has_slice_parm) {
            slice_spec = slice_spec_in;
        }
        return this->call_with_parms(F, g, slice_spec, std::make_index_sequence<TypicalOpIO<Ftype>::parm_n_inout>{});
    }

#ifdef PREPARE_DISABLED
    /**
     * @brief We can reduce the loading time for the no-prepare skel by providing flags directly
     * with a function, rather than by populating the global op_info_map at static-init time.
     */
    virtual Flags_word get_flag_word() const override { return hnnx::flags_for<TypicalOp>; }
#endif
    virtual const char *get_docs() const override { return hnnx::docs_for<TypicalOp>(); }

  protected:
    TypicalOp(OpIoPtrs const &ioptrs) : TypicalOpIO<Ftype>(ioptrs) {}
};

/** Generate and register an op with Derived type and Function Type
 *  Use this for other Derived Ops similar to const Op
 */

template <auto F, typename OpaqueT> class TypicalOpWithCompiler : public TypicalOp<F> {
    // an error occurs here if you failed to specialize op_opaque_tgt_info for your OpaqueT
    static constexpr unsigned tgt_opaque_size = op_opaque_tgt_info<OpaqueT>::length;
    static constexpr unsigned tgt_opaque_align = op_opaque_tgt_info<OpaqueT>::alignment;
    // when compiled on hexagon; this throws a static_assert if the size/align are wrong.
    static constexpr bool check_szal0 =
            bake::check_size_align<OpaqueT>::template check<tgt_opaque_size, tgt_opaque_align>();
    static_assert(check_szal0);
    // LCOV_EXCL_START [SAFTYSWCCB-1735] compile-time static_assert helper has no runtime behavior
    static inline constexpr bool check_szal()
    {
        constexpr auto claimed_size = bake::typical_op_extra_tgt_size_align(
                TypicalOp<F>::n_inputs, TypicalOp<F>::n_outputs, tgt_opaque_size, tgt_opaque_align);
        return bake::check_size_align<TypicalOpWithCompiler<F, OpaqueT>>::template check<std::get<0>(claimed_size),
                                                                                         std::get<1>(claimed_size)>();
    }
    // LCOV_EXCL_STOP

  public:
    using ThisType = TypicalOpWithCompiler<F, OpaqueT>;
    using Parent = TypicalOp<F>;
    mutable OpaqueT opaque{};
    explicit TypicalOpWithCompiler(Deserz &dctx) : Parent(dctx) {}
    virtual Executable::ItemType compile(Graph &graph_in) const override;
    virtual bool check_constraint_for_recompile(Graph &graph_in) const override;
    // Generator function.  Create the op if possible, otherwise return null.
    static uptr_Op create(OpIoPtrs const &ioptrs, const OpId my_id_in)
    {
        // Make sure we can create it.
        if (!ThisType::valid_construction(ioptrs, my_id_in)) return uptr_Op{};
        ThisType *op = new ThisType{ioptrs};
        // Hook in the inputs/outputs early so we can use cost()
        op->assign_input_output_pointers(ioptrs);
        return uptr_Op{op};
    }

#ifdef PREPARE_DISABLED
    /** @see hnnx::TypicalOp::get_flag_word() */
    virtual Flags_word get_flag_word() const override { return hnnx::flags_for<TypicalOpWithCompiler>; }
#else
    virtual const char *get_docs() const override { return hnnx::docs_for<TypicalOpWithCompiler>(); }
    // This does almost the same thing as over-ridden TypicalOpIoBase<N_OUT,N_IN>::serialize,
    // but also provides size/align info about OpaqueT to the serializer.
    virtual void serialize(SerOpsInterface &sctx) const override
    {
        sctx.op_typical_with_extra<OpaqueT>(this, this->io.inputs(), this->io.outputs());
    }
#endif
  protected:
    TypicalOpWithCompiler(OpIoPtrs const &ioptrs) : Parent(ioptrs) {}
};

} // namespace hnnx

template <auto F> constexpr bool has_compile_method = false;

template <auto F> struct API_HIDDEN OpaqueT_FOR {
    using type = char;
};

template <auto F> using FType_of = typename std::remove_pointer<decltype(F)>::type;

template <auto F> struct API_HIDDEN DerivedType {
    using FType = FType_of<F>;
    using type = std::conditional_t<has_compile_method<F>,
                                    hnnx::TypicalOpWithCompiler<F, typename OpaqueT_FOR<F>::type>, hnnx::TypicalOp<F>>;
};

namespace hnnx {

template <typename OpT> API_HIDDEN inline constexpr unsigned get_op_num_inputs()
{
    return OpT::n_inputs;
}
template <typename OpT> API_HIDDEN inline constexpr unsigned get_op_num_outputs()
{
    return OpT::n_outputs;
}

template <typename OpT, unsigned IDX>
using op_input_t = std::remove_pointer_t<typename std::tuple_element_t<IDX, typename OpT::input_tuple_type>>;
template <typename OpT, unsigned IDX>
using op_output_t = std::remove_pointer_t<typename std::tuple_element_t<IDX, typename OpT::output_tuple_type>>;

// do the 'constructor hook' operations for a given placement op,
// where flat output [O_IDX] is to be allocated as flat input I_IDX.

template <unsigned IDX, typename OpT> API_HIDDEN inline auto const &get_op_input(OpT &op)
{
    using in_ttype = op_input_t<OpT, IDX>;
    in_ttype const &in = *static_cast<in_ttype const *>(op.io.inputs()[IDX]);
    return in;
}
template <unsigned IDX, typename OpT> API_HIDDEN inline auto &get_op_output(OpT &op)
{
    using out_ttype = op_output_t<OpT, IDX>;
    out_ttype &out = *static_cast<out_ttype *>(op.io.outputs()[IDX].get());
    return out;
}

template <unsigned O_IDX, unsigned I_IDX, typename OpT> void ctor_hook_flat_inplace(Graph &, OpT &op)
{
    auto &out = get_op_output<O_IDX>(op);
    auto const &in = get_op_input<I_IDX>(op);
    out.data_ptr() = in.data_ptr();
}

template <unsigned O_IDX, unsigned I_IDX, typename OpT> void ctor_hook_crouton_inplace(Graph &, OpT &op)
{
    auto &out = get_op_output<O_IDX>(op);
    auto const &in = get_op_input<I_IDX>(op);
    memcpy(out.blocktab_ptr(), in.blocktab_ptr(), out.blocktab_len() * sizeof(void *));
}

} //namespace hnnx

POP_VISIBILITY()

#include "op_register.h"
#include "cost.h"

#endif
