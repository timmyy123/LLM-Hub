#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// LLMHub: byte offset in dst of flat element n. Three divisions instead of seven, and 32-bit ones
// whenever n fits (64-bit division is emulated on Adreno and much slower).
inline ulong llmhub_cpy_dst_offset(ulong n, int ne0, int ne1, int ne2, ulong nb0, ulong nb1, ulong nb2, ulong nb3) {
    if (n <= 0xFFFFFFFFul) {
        const uint m  = (uint) n;
        const uint q0 = m  / (uint) ne0;
        const uint q1 = q0 / (uint) ne1;
        const uint q2 = q1 / (uint) ne2;
        return (m - q0*(uint) ne0)*nb0 + (q0 - q1*(uint) ne1)*nb1 + (q1 - q2*(uint) ne2)*nb2 + q2*nb3;
    }
    const ulong q0 = n  / ne0;
    const ulong q1 = q0 / ne1;
    const ulong q2 = q1 / ne2;
    return (n - q0*ne0)*nb0 + (q0 - q1*ne1)*nb1 + (q1 - q2*ne2)*nb2 + q2*nb3;
}

inline ulong llmhub_cpy_i0(ulong n, int ne0) {
    return n <= 0xFFFFFFFFul ? (ulong) ((uint) n % (uint) ne0) : n % ne0;
}

//------------------------------------------------------------------------------
// cpy
//------------------------------------------------------------------------------

kernel void kernel_cpy_f16_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = (((ulong)i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        global half * out = (global half *)((global char *) dst + dst_off);
        out[0] = src[0];
    }
}

kernel void kernel_cpy_f16_f32(
        global half * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = (((ulong)i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        global float * out = (global float *)((global char *) dst + dst_off);
        out[0] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        global float * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = (((ulong)i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        global half * out = (global half *)((global char *) dst + dst_off);
        out[0] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = (((ulong)i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        global float * out = (global float *)((global char *) dst + dst_off);
        out[0] = src[0];
    }
}

#ifndef GGML_CL_NO_CPY_PACK
kernel void kernel_cpy_f32_f32_pack(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int lsz = get_local_size(0);
    int tpr = min(ne00, lsz);          // threads per row
    int rpw = lsz / tpr;               // rows per workgroup
    int lid = get_local_id(0);
    int row = get_group_id(0)*rpw + lid / tpr;
    int lane = lid - (lid / tpr) * tpr;

    int nrows = ne01*ne02*ne03;
    if (row >= nrows) {
        return;
    }

    int i01 = row % ne01;
    int t   = row / ne01;
    int i02 = t % ne02;
    int i03 = t / ne02;

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = (ulong)row * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (int i00 = lane; i00 < ne00; i00 += tpr) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        global float * out = (global float *)((global char *) dst + dst_off);
        out[0] = src[0];
    }
}
#endif // GGML_CL_NO_CPY_PACK

kernel void kernel_cpy_i32_i32(
        global int * src0,
        ulong offset0,
        global int * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global int*)((global char*)src0 + offset0);
    dst = (global int*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = (((ulong)i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const int * src = (global int *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        global int * out = (global int *)((global char *) dst + dst_off);
        out[0] = src[0];
    }
}

// Contiguous f32 copy, one work item per float4 over the whole tensor. The kernels above map
// one workgroup to each row, which leaves a tensor with few long rows on a single compute unit.
// vload4/vstore4 rather than a float4 cast: these buffers carry an arbitrary 4-byte view offset.
kernel void kernel_cpy_f32_f32_flat(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        ulong ne,
        ulong n4
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst  + offsetd);

    const ulong i = get_global_id(0);

    if (i < n4) {
        vstore4(vload4(i, src0), i, dst);
    } else if (i == n4) {
        for (ulong t = n4 * 4; t < ne; ++t) {
            dst[t] = src0[t];
        }
    }
}

kernel void kernel_cpy_i32_f32(
        global int * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global int *) ((global char *) src0 + offset0);
    dst  = (global float *) ((global char *) dst  + offsetd);

    const ulong i01 = get_group_id(0);
    const ulong i02 = get_group_id(1);
    const ulong i03 = get_group_id(2);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = ((i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (ulong i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const int * src = (global const int *) ((global char *) src0 + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
        global float * out = (global float *) ((global char *) dst + dst_off);
        out[0] = convert_float(src[0]);
    }
}


kernel void kernel_cpy_f32_i32(
        global float * src0,
        ulong offset0,
        global int * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float *) ((global char *) src0 + offset0);
    dst  = (global int *) ((global char *) dst  + offsetd);

    const ulong i01 = get_group_id(0);
    const ulong i02 = get_group_id(1);
    const ulong i03 = get_group_id(2);

    // LLMHub: locate the start of this src row in dst once; per-element index math made these copies
    // several times slower on Adreno. Only a src row that spans dst rows is located per element.
    const ulong n_row    = ((i03 * ne02 + i02) * ne01 + i01) * ne00;
    const ulong row_off  = llmhub_cpy_dst_offset(n_row, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
    const bool  row_fits = llmhub_cpy_i0(n_row, ne0) + ne00 <= (ulong) ne0;

    for (ulong i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        const ulong dst_off = row_fits ? row_off + i00*nb0
                                       : llmhub_cpy_dst_offset(n_row + i00, ne0, ne1, ne2, nb0, nb1, nb2, nb3);

        global const float * src = (global const float *) ((global char *) src0 + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
        global int * out = (global int *) ((global char *) dst + dst_off);
        out[0] = convert_int_rtz(src[0]);
    }
}
