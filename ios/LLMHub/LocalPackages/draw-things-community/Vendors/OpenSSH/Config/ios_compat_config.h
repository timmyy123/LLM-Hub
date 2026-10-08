#ifndef LOCAL_CODE_OPENSSH_IOS_COMPAT_CONFIG_H
#define LOCAL_CODE_OPENSSH_IOS_COMPAT_CONFIG_H

/*
 * OpenSSH's configure probes use unprefixed libcrypto symbols. SwiftCrypto's
 * BoringSSL intentionally prefixes every exported symbol, so record the APIs
 * available in the pinned BoringSSL revision here.
 */
#define HAVE_BN_IS_PRIME_EX 1
#define HAVE_EC_POINT_GET_AFFINE_COORDINATES 1
#define HAVE_EC_POINT_GET_AFFINE_COORDINATES_GFP 1
#define HAVE_EC_POINT_SET_AFFINE_COORDINATES 1
#define HAVE_EC_POINT_SET_AFFINE_COORDINATES_GFP 1
#define HAVE_EVP_DIGESTFINAL_EX 1
#define HAVE_EVP_DIGESTINIT_EX 1
#define HAVE_EVP_DIGESTSIGN 1
#define HAVE_EVP_DIGESTVERIFY 1
#define HAVE_EVP_MD_CTX_CLEANUP 1
#define HAVE_EVP_MD_CTX_COPY_EX 1
#define HAVE_EVP_MD_CTX_INIT 1
#define HAVE_EVP_PKEY_GET_RAW_PRIVATE_KEY 1
#define HAVE_EVP_PKEY_GET_RAW_PUBLIC_KEY 1
#define HAVE_EVP_SHA256 1
#define HAVE_EVP_SHA384 1
#define HAVE_EVP_SHA512 1
#define HAVE_RSA_GENERATE_KEY_EX 1

#include <TargetConditionals.h>

#if TARGET_OS_IPHONE
/* Use ios_system's controlling terminal instead of /dev/tty. */
#undef HAVE_READPASSPHRASE
#undef HAVE_READPASSPHRASE_H

/* These macOS facilities are unavailable or inappropriate in an iOS app. */
#undef HAVE_DAEMON
#undef HAVE_GETTTYENT
#undef HAVE_LIBPROC_H
#undef HAVE_LOGWTMP
#undef HAVE_NLIST
#undef HAVE_NLIST_H
#undef HAVE_SANDBOX_H
#undef HAVE_SANDBOX_INIT
#undef HAVE_SETPASSENT
#undef HAVE_SETLOGIN
#undef HAVE_SYS_PTRACE_H
#undef HAVE_SYS_RANDOM_H
#undef HAVE_TTYENT_H
#undef HAVE_UTMP_H
#endif

#endif
