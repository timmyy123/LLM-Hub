#include "memory_topology.h"

#import <Metal/Metal.h>

#include <cstring>

namespace bmoe::detail {

Tri metal_memory_sharing(const char * description) {
    if (!description || !*description) return Tri::Unknown;
    @autoreleasepool {
        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || std::strcmp(description, device.name.UTF8String) != 0) return Tri::Unknown;
            return device.hasUnifiedMemory ? Tri::Yes : Tri::No;
        }
    }
    return Tri::Unknown;
}

} // namespace bmoe::detail
