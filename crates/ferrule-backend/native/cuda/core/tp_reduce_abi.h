#ifndef FERRULE_CUDA_CORE_TP_REDUCE_ABI_H_
#define FERRULE_CUDA_CORE_TP_REDUCE_ABI_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Opaque host handles; Ferrule retains the exact context/stream and backing.
typedef struct FerruleTpLibrary FerruleTpLibrary;
typedef struct FerruleTpComm FerruleTpComm;
typedef struct FerruleTpUniqueId { uint8_t bytes[128]; } FerruleTpUniqueId;
enum { FERRULE_TP_OK = 0, FERRULE_TP_INVALID = 1,
       FERRULE_TP_UNAVAILABLE = 2, FERRULE_TP_PENDING = 3,
       FERRULE_TP_UNKNOWN = 9 };
size_t ferrule_tp_library_host_bytes(void);
size_t ferrule_tp_comm_host_bytes(void);
int32_t ferrule_tp_library_open(const char *absolute_path, FerruleTpLibrary **out);
void ferrule_tp_library_close(FerruleTpLibrary *library);
int32_t ferrule_tp_unique_id(FerruleTpLibrary *library, FerruleTpUniqueId *out);
// Nonblocking init. A non-null out always owns custody, including error returns.
int32_t ferrule_tp_comm_create(FerruleTpLibrary *library, const FerruleTpUniqueId *id,
    uint32_t rank, uint64_t stream, uint64_t storage, uint64_t storage_bytes,
    FerruleTpComm **out);
int32_t ferrule_tp_comm_async_status(FerruleTpComm *comm);
int32_t ferrule_tp_comm_warmup(FerruleTpComm *comm);
int32_t ferrule_tp_allgather_bf16_two(FerruleTpComm *comm,
    uint64_t input, uint64_t input_bytes, uint32_t count);
// One-shot progress of an in-progress host enqueue, then ordered addition.
int32_t ferrule_tp_comm_progress(FerruleTpComm *comm);
// Caller must prove its recorded compute event complete first. Copies FOUR bytes.
int32_t ferrule_tp_comm_read_status(FerruleTpComm *comm, uint32_t *out);
// Abort is NOT CUDA/NCCL completion proof; handle and storage stay in custody.
int32_t ferrule_tp_comm_abort(FerruleTpComm *comm);
// Only after positive stream completion and successful async status, never Unknown.
int32_t ferrule_tp_comm_destroy(FerruleTpComm *comm);
#ifdef __cplusplus
}
#endif
#endif
