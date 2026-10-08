#pragma once

// Lean calls mi_* explicitly. Never replace or interpose the host allocator,
// including through an Apple malloc zone. Defined-as-zero also enables some
// upstream override paths, so reject the presence of these macros entirely.
#if defined(MI_MALLOC_OVERRIDE) || defined(MI_OSX_INTERPOSE) || \
    defined(MI_OSX_ZONE)
#error "Embedded Lean must not override the host allocator"
#endif
