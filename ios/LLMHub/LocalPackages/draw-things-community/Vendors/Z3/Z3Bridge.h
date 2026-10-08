#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Z3BridgeStatus {
  Z3BridgeStatusSat = 0,
  Z3BridgeStatusUnsat = 1,
  Z3BridgeStatusUnknown = 2,
  Z3BridgeStatusTimeout = 3,
  Z3BridgeStatusError = 4,
} Z3BridgeStatus;

typedef struct Z3BridgeResult {
  Z3BridgeStatus status;
  char *model;
  char *reason;
} Z3BridgeResult;

Z3BridgeResult Z3BridgeCheckSMT(const char *code, uint32_t timeoutMilliseconds);
void Z3BridgeFreeResult(Z3BridgeResult result);

#ifdef __cplusplus
}
#endif
