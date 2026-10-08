#include "config.h"

#if TARGET_OS_IPHONE
#include <openbsd-compat/readpassphrase.h>
#else
#include_next <readpassphrase.h>
#endif
