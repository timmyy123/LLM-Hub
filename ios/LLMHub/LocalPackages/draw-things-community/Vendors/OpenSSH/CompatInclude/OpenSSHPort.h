#ifndef LOCAL_CODE_OPENSSH_PORT_H
#define LOCAL_CODE_OPENSSH_PORT_H

// ios_error.h redirects stdio/process calls into ios_system. Import the wide
// character declarations first so its compatibility macros do not rewrite
// declarations in the Apple SDK headers.
#include <wchar.h>
#include <wctype.h>

#include "ios_error.h"

#endif
