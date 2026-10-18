// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#pragma once
namespace hnnx {
enum {
    PMU_CYCLES = 0,
    PMU_PKT_USR = 2,
    PMU_PKT_GST = 5,
    PMU_PKT_MON = 6,
    PMU_QTIMER_NS = 21,
    PMU_NUM_32B_CTRS = 8,
    PMU_NUM_64B_CTRS = 4,
    PMU_FIRST_SOFTWARE_EVENT = 1024, // The following are implemented in software:
    PMU_DMA_WAIT = 1024,
    PMU_HVX_WAIT = 1025,
    PMU_DESER = 1026,
    PMU_COMPRESSION_ENABLED = 1027,
    PMU_BYTES_COMPRESSED = 1028,
    PMU_COMPRESSED_BYTES_WRITTEN = 1029,
    PMU_AFTER_SOFTWARE_EVENTS,
    PMU_NUM_SOFTWARE_EVENTS = PMU_AFTER_SOFTWARE_EVENTS - PMU_FIRST_SOFTWARE_EVENT
};
}
