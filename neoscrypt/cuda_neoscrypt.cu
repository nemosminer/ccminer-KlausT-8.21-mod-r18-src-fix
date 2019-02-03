// originally from djm34 - github.com/djm34/ccminer-sp-neoscrypt
// kernel code from Nanashi Meiyo-Meijin 1.7.6-r10 (July 2016)
// modified by tpruvot

#include <stdio.h>
#include <memory.h>
#include "cuda_helper.h"
#include "cuda_vector.h" 
#include "miner.h"

#ifdef _MSC_VER
#define THREAD __declspec(thread)
#else
#define THREAD __thread
#endif

#define rotate ROTL32
#define rotateR ROTR32
#define rotateL ROTL32

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 520
#if __CUDA_ARCH__ >= 320
__device__ uint32_t __funnelshift_lc(uint32_t lo, uint32_t hi, uint32_t shift);
__device__ ​uint32_t __funnelshift_rc(uint32_t lo, uint32_t hi, uint32_t shift);
#endif
#endif
#if __CUDA_ARCH__ < 320
#define __funnelshift_lc(lo, hi, shift) (((lo) >> (32 - (shift))) | ((hi) << (shift)))
#define __funnelshift_rc(lo, hi, shift) (((hi) << (32 - (shift))) | ((lo) >> (shift)))
#define __ldg(x) (*(x))
#define __ldg4(x) (*(x))
#endif
#if defined(CUDART_VERSION) && CUDART_VERSION < 9000
#define __syncwarp(mask) __threadfence_block()
#endif

#define SHIFT 128U

#define TPB60 64
#define TPB52 128
#define TPB50 128
#define TPB30 192
#define TPB20 96

#if __CUDA_ARCH__ >= 700
#define TPB TPB60
#define BPM1 8
#define BPM2 4
#elif __CUDA_ARCH__ >= 610
#define TPB TPB52
#define BPM1 4
#define BPM2 3
#elif __CUDA_ARCH__ >= 600
#define TPB TPB60
#define BPM1 8
#define BPM2 4
#elif __CUDA_ARCH__ >= 520
#define TPB TPB52
#define BPM1 4
#define BPM2 3
#elif __CUDA_ARCH__ >= 500
#define TPB TPB50
#define BPM1 4
#define BPM2 2
#elif __CUDA_ARCH__ >= 300
#define TPB TPB30
#define BPM1 3
#define BPM2 1
#else
#define TPB TPB20
#define BPM1 3
#define BPM2 2
#endif

static uint32_t* d_NNonce[MAX_GPUS];

__device__ uint4* W;
__device__ uint4* Tr;
__device__ uint4* Tr2;
__device__ uint4* Input;

__constant__ uint32_t c_data[64];
__constant__ uint32_t c_target[2];
__constant__ uint32_t key_init[16];
__constant__ uint32_t input_init[16];

static const __constant__ uint8 BLAKE2S_IV_Vec = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint8 BLAKE2S_IV_Vechost = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

__constant__ uint32_t BLAKE2S_SIGMA[10][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

#define BLOCK_SIZE         64U
#define BLAKE2S_BLOCK_SIZE 64U
#define BLAKE2S_OUT_SIZE   32U

#if __CUDA_ARCH__ < 300
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;

	shared_mem[TPB * 0 + thread] = a;
	__syncwarp(0xFFFFFFFF);
	return shared_mem[0 * TPB + bitselect(thread, b, c - 1)];
}

__device__ __forceinline__ void WarpShuffle3(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;

	shared_mem[TPB * 0 + thread] = a0;
	shared_mem[TPB * 1 + thread] = a1;
	shared_mem[TPB * 2 + thread] = a2;
	__syncwarp(0xFFFFFFFF);
	d0 = shared_mem[0 * TPB + bitselect(thread, b0, c - 1)];
	d1 = shared_mem[1 * TPB + bitselect(thread, b1, c - 1)];
	d2 = shared_mem[2 * TPB + bitselect(thread, b2, c - 1)];
}

#else
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return SHFL(a, b, c);
}

__device__ __forceinline__ void WarpShuffle3(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
	d2 = WarpShuffle(a2, b2, c);
}

#endif

static __device__ __forceinline__ uint4 __ldL1(const uint4 *ptr)
{
	uint4 ret;
	asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint4 *ptr, const uint4 value)
{
	asm("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" ::__LDG_PTR(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

static __device__ __forceinline__ uint32_t __ldL1(const uint32_t *ptr)
{
	uint32_t ret;
	asm("ld.global.ca.u32 %0, [%1];" : "=r"(ret) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint32_t *ptr, const uint32_t value)
{
	asm("st.global.wb.u32 [%0], %1;" ::__LDG_PTR(ptr), "r"(value));
}

#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	idx = BLAKE2S_SIGMA[idx0][(idx1)+1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE(a, b, c, d, key1,key2) { \
	a += key1; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key2; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE(idx0,idx1, a, b, c, d, key) { \
	a += key[idx0]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key[idx1]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE0(idx0,idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE1(idx0,idx1, a, b, c, d, key) { \
	a += key[idx0]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE2(idx0,idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key[idx1]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

static __forceinline__ __device__
void Blake2S(uint32_t *out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey)
{
	uint16 V;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	// { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	// { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	// { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	// { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[9], inout[0]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[5], inout[7]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[2], inout[4]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[10], inout[15]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[14], inout[1]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[11], inout[12]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[6], inout[8]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[3], inout[13]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[2], inout[12]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[6], inout[10]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[0], inout[11]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[8], inout[3]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[4], inout[13]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[7], inout[5]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[15], inout[14]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[1], inout[9]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[12], inout[5]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[1], inout[15]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[14], inout[13]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[4], inout[10]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[0], inout[7]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[6], inout[3]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[9], inout[2]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[8], inout[11]);
	// 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[13], inout[11]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[7], inout[14]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[12], inout[1]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[3], inout[9]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[5], inout[0]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[15], inout[4]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[8], inout[6]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[2], inout[10]);
	// 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[6], inout[15]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[14], inout[9]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[11], inout[3]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[0], inout[8]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[12], inout[2]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[13], inout[7]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[1], inout[4]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[10], inout[5]);
	// 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[10], inout[2]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[8], inout[4]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[7], inout[6]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[1], inout[5]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[15], inout[11]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[9], inout[14]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[3], inout[12]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[13], inout[0]);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	((uint8*)out)[0] = V.lo;
}

#define SALSA(a,b,c,d) { \
	uint32_t t; \
	t = rotateL(a+d,  7U); b ^= t; \
	t = rotateL(b+a,  9U); c ^= t; \
	t = rotateL(c+b, 13U); d ^= t; \
	t = rotateL(d+c, 18U); a ^= t; \
}

#define CHACHA_STEP(a,b,c,d) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateL(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x2103); \
	c += d; b = rotateL(b^c, 7); \
}

#define SALSA_CORE(state) { \
	SALSA(state[0], state[4], state[8], state[12]); \
	SALSA(state[5], state[9], state[13], state[1]); \
	SALSA(state[10], state[14], state[2], state[6]); \
	SALSA(state[15], state[3], state[7], state[11]); \
	SALSA(state[0], state[1], state[2], state[3]); \
	SALSA(state[5], state[6], state[7], state[4]); \
	SALSA(state[10], state[11], state[8], state[9]); \
	SALSA(state[15], state[12], state[13], state[14]); \
}

#define CHACHA_CORE(state)	{ \
	CHACHA_STEP(state[0], state[4], state[8], state[12]); \
	CHACHA_STEP(state[1], state[5], state[9], state[13]); \
	CHACHA_STEP(state[2], state[6], state[10], state[14]); \
	CHACHA_STEP(state[3], state[7], state[11], state[15]); \
	CHACHA_STEP(state[0], state[5], state[10], state[15]); \
	CHACHA_STEP(state[1], state[6], state[11], state[12]); \
	CHACHA_STEP(state[2], state[7], state[8], state[13]); \
	CHACHA_STEP(state[3], state[4], state[9], state[14]); \
}

#define SALSA_CORE_PARALLEL(state) { \
	SALSA(state.x, state.y, state.z, state.w); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4); \
	SALSA(state.x, state.w, state.z, state.y); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4); \
}

#define CHACHA_CORE_PARALLEL(state)	{ \
	CHACHA_STEP(state.x, state.y, state.z, state.w); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4); \
	CHACHA_STEP(state.x, state.y, state.z, state.w); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4); \
}

__forceinline__ __device__
void salsa_small_rnd(uint4 X[4])
{
	uint32_t state[16];
	((uint4*)state)[0] = X[0];
	((uint4*)state)[1] = X[1];
	((uint4*)state)[2] = X[2];
	((uint4*)state)[3] = X[3];

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		SALSA_CORE(state);
	}

	X[0] += ((uint4*)state)[0];
	X[1] += ((uint4*)state)[1];
	X[2] += ((uint4*)state)[2];
	X[3] += ((uint4*)state)[3];
}

__device__ __forceinline__
void chacha_small_rnd(uint4 X[4])
{
	uint32_t state[16];
	((uint4*)state)[0] = X[0];
	((uint4*)state)[1] = X[1];
	((uint4*)state)[2] = X[2];
	((uint4*)state)[3] = X[3];

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		CHACHA_CORE(state);
	}

	X[0] += ((uint4*)state)[0];
	X[1] += ((uint4*)state)[1];
	X[2] += ((uint4*)state)[2];
	X[3] += ((uint4*)state)[3];
}

__forceinline__ __device__
uint4 salsa_small_parallel_rnd(uint4 X)
{
	uint4 state = X;

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		SALSA_CORE_PARALLEL(state);
	}

	return (X + state);
}

__device__ __forceinline__
uint4 chacha_small_parallel_rnd(uint4 X)
{
	uint4 state = X;

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		CHACHA_CORE_PARALLEL(state);
	}

	return (X + state);
}

__device__ __forceinline__
void neoscrypt_chacha(uint4 XV[16])
{
	uint4 temp[4];

	XV[0] ^= XV[12];
	XV[1] ^= XV[13];
	XV[2] ^= XV[14];
	XV[3] ^= XV[15];
	chacha_small_rnd(&XV[0]);
	temp[0] = XV[4] ^ XV[0];
	temp[1] = XV[5] ^ XV[1];
	temp[2] = XV[6] ^ XV[2];
	temp[3] = XV[7] ^ XV[3];
	chacha_small_rnd(temp);
	XV[4] = XV[8] ^ temp[0];
	XV[5] = XV[9] ^ temp[1];
	XV[6] = XV[10] ^ temp[2];
	XV[7] = XV[11] ^ temp[3];
	chacha_small_rnd(&XV[4]);
	XV[12] ^= XV[4];
	XV[13] ^= XV[5];
	XV[14] ^= XV[6];
	XV[15] ^= XV[7];
	chacha_small_rnd(&XV[12]);
	XV[8] = temp[0];
	XV[9] = temp[1];
	XV[10] = temp[2];
	XV[11] = temp[3];
}

__device__ __forceinline__
void neoscrypt_salsa(uint4 XV[16])
{
	uint4 temp[4];

	XV[0] ^= XV[12];
	XV[1] ^= XV[13];
	XV[2] ^= XV[14];
	XV[3] ^= XV[15];
	salsa_small_rnd(&XV[0]);
	temp[0] = XV[4] ^ XV[0];
	temp[1] = XV[5] ^ XV[1];
	temp[2] = XV[6] ^ XV[2];
	temp[3] = XV[7] ^ XV[3];
	salsa_small_rnd(temp);
	XV[4] = XV[8] ^ temp[0];
	XV[5] = XV[9] ^ temp[1];
	XV[6] = XV[10] ^ temp[2];
	XV[7] = XV[11] ^ temp[3];
	salsa_small_rnd(&XV[4]);
	XV[12] ^= XV[4];
	XV[13] ^= XV[5];
	XV[14] ^= XV[6];
	XV[15] ^= XV[7];
	salsa_small_rnd(&XV[12]);
	XV[8] = temp[0];
	XV[9] = temp[1];
	XV[10] = temp[2];
	XV[11] = temp[3];
}

__device__ __forceinline__
void neoscrypt_chacha_parallel(uint4 XV[4])
{
	uint4 temp;

	XV[0] = chacha_small_parallel_rnd(XV[0] ^ XV[3]);
	temp = chacha_small_parallel_rnd(XV[1] ^ XV[0]);
	XV[1] = chacha_small_parallel_rnd(XV[2] ^ temp);
	XV[3] = chacha_small_parallel_rnd(XV[3] ^ XV[1]);
	XV[2] = temp;
}

__device__ __forceinline__
void neoscrypt_salsa_parallel(uint4 XV[4])
{
	uint4 temp;

	XV[0] = salsa_small_parallel_rnd(XV[0] ^ XV[3]);
	temp =salsa_small_parallel_rnd(XV[1] ^ XV[0]);
	XV[1] = salsa_small_parallel_rnd(XV[2] ^ temp);
	XV[3] = salsa_small_parallel_rnd(XV[3] ^ XV[1]);
	XV[2] = temp;
}

static __forceinline__ __device__
void fastkdf256(const uint32_t thread, const uint32_t nonce)
{
	extern __shared__ uint32_t B[];

	uint32_t bufidx, qbuf, rbuf, bitbuf;
	uint32_t shift, a, b, noncepos;
	uint32_t input[16];
	uint32_t key[16];
	uint32_t temp[9];

	for (int i = 0; i < 64; i++)
		B[i * TPB + threadIdx.x] = c_data[i];

	B[19 * TPB + threadIdx.x] = nonce;
	B[39 * TPB + threadIdx.x] = nonce;
	B[59 * TPB + threadIdx.x] = nonce;

#pragma unroll
	for (int i = 0; i < 16; i++) input[i] = input_init[i];
#pragma unroll
	for (int i = 0; i < 8; i++) key[i] = key_init[i];
#pragma unroll
	for (int i = 8; i < 16; i++) key[i] = 0;

#pragma unroll 1
	for (int i = 0; i < 31; i++)
	{
		bufidx = 0;
#pragma unroll
		for (int x = 0; x < 8; ++x)
			bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
		bufidx += bufidx >> 16;
		bufidx &= 0x000000ff;
		qbuf = bufidx >> 2;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;

		shift = 32U - bitbuf;
		temp[0] = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] ^ (input[0] << bitbuf);
		temp[1] = B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[0], input[1], shift);
		temp[2] = B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[1], input[2], shift);
		temp[3] = B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[2], input[3], shift);
		temp[4] = B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[3], input[4], shift);
		temp[5] = B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[4], input[5], shift);
		temp[6] = B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[5], input[6], shift);
		temp[7] = B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[6], input[7], shift);
		temp[8] = B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] ^ (input[7] >> shift);

		B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] = temp[0];
		B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] = temp[1];
		B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] = temp[2];
		B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] = temp[3];
		B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] = temp[4];
		B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] = temp[5];
		B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] = temp[6];
		B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] = temp[7];
		B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] = temp[8];

		noncepos = qbuf < 20U ? 19U : 39U;
		noncepos = qbuf < 40U ? noncepos : 59U;

		a = (qbuf + 0) & 0x3f;
		a = a == noncepos ? nonce : c_data[a];

#pragma unroll
		for (int k = 0; k < 16; k += 2)
		{
			b = (qbuf + k + 1) & 0x3f;
			b = b == noncepos ? nonce : c_data[b];
			input[k + 0] = __funnelshift_rc(a, b, bitbuf);

			a = (qbuf + k + 2) & 0x3f;
			a = a == noncepos ? nonce : c_data[a];
			input[k + 1] = __funnelshift_rc(b, a, bitbuf);
		}

		key[0] = __funnelshift_rc(temp[0], temp[1], bitbuf);
		key[1] = __funnelshift_rc(temp[1], temp[2], bitbuf);
		key[2] = __funnelshift_rc(temp[2], temp[3], bitbuf);
		key[3] = __funnelshift_rc(temp[3], temp[4], bitbuf);
		key[4] = __funnelshift_rc(temp[4], temp[5], bitbuf);
		key[5] = __funnelshift_rc(temp[5], temp[6], bitbuf);
		key[6] = __funnelshift_rc(temp[6], temp[7], bitbuf);
		key[7] = __funnelshift_rc(temp[7], temp[8], bitbuf);

		Blake2S(input, input, key);
	}

	bufidx = 0;
#pragma unroll
	for (int x = 0; x < 8; ++x)
		bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
	bufidx += bufidx >> 16;
	bufidx &= 0x000000ff;
	qbuf = bufidx >> 2;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	a = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x];
#pragma unroll
	for (int i = 0; i < 8; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = __funnelshift_rc(a, b, bitbuf) ^ input[i + 0] ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = __funnelshift_rc(b, a, bitbuf) ^ input[i + 1] ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = __funnelshift_rc(a, b, bitbuf) ^ input[i + 2] ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = __funnelshift_rc(b, a, bitbuf) ^ input[i + 3] ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
#pragma unroll
	for (int i = 8; i < 16; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = __funnelshift_rc(b, a, bitbuf) ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
	b = B[((qbuf + 17) & 0x3f) * TPB + threadIdx.x];
	temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[16];
	a = B[((qbuf + 18) & 0x3f) * TPB + threadIdx.x];
	temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[17];
	b = B[((qbuf + 19) & 0x3f) * TPB + threadIdx.x];
	temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[18];
	a = B[((qbuf + 20) & 0x3f) * TPB + threadIdx.x];
	temp[3] = __funnelshift_rc(b, a, bitbuf) ^ nonce;
	Input[16U * thread + 4] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
#pragma unroll
	for (int i = 20; i < 36; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = __funnelshift_rc(b, a, bitbuf) ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
	b = B[((qbuf + 37) & 0x3f) * TPB + threadIdx.x];
	temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[36];
	a = B[((qbuf + 38) & 0x3f) * TPB + threadIdx.x];
	temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[37];
	b = B[((qbuf + 39) & 0x3f) * TPB + threadIdx.x];
	temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[38];
	a = B[((qbuf + 40) & 0x3f) * TPB + threadIdx.x];
	temp[3] = __funnelshift_rc(b, a, bitbuf) ^ nonce;
	Input[16U * thread + 9] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
#pragma unroll
	for (int i = 40; i < 56; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = __funnelshift_rc(b, a, bitbuf) ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
	b = B[((qbuf + 57) & 0x3f) * TPB + threadIdx.x];
	temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[56];
	a = B[((qbuf + 58) & 0x3f) * TPB + threadIdx.x];
	temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[57];
	b = B[((qbuf + 59) & 0x3f) * TPB + threadIdx.x];
	temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[58];
	a = B[((qbuf + 60) & 0x3f) * TPB + threadIdx.x];
	temp[3] = __funnelshift_rc(b, a, bitbuf) ^ nonce;
	Input[16U * thread + 14] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	b = B[((qbuf + 61) & 0x3f) * TPB + threadIdx.x];
	temp[0] = __funnelshift_rc(a, b, bitbuf) ^ c_data[60];
	a = B[((qbuf + 62) & 0x3f) * TPB + threadIdx.x];
	temp[1] = __funnelshift_rc(b, a, bitbuf) ^ c_data[61];
	b = B[((qbuf + 63) & 0x3f) * TPB + threadIdx.x];
	temp[2] = __funnelshift_rc(a, b, bitbuf) ^ c_data[62];
	a = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x];
	temp[3] = __funnelshift_rc(b, a, bitbuf) ^ c_data[63];
	Input[16U * thread + 15] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
}

static __forceinline__ __device__
uint32_t fastkdf32(uint32_t thread, const uint32_t nonce)
{
	extern __shared__ uint32_t B[];

	uint32_t bufidx, qbuf, rbuf, bitbuf;
	uint32_t shift, a, b, noncepos;
	uint32_t input[16];
	uint32_t key[16];
	uint32_t temp[9];

#pragma unroll
	for (int i = 0; i < 16; i++) {
		((uint4*)key)[0] = __ldg(Tr2 + 16U * thread + i) ^ __ldg(Tr + 16U * thread + i);
		B[(i * 4 + 0) * TPB + threadIdx.x] = key[0];
		B[(i * 4 + 1) * TPB + threadIdx.x] = key[1];
		B[(i * 4 + 2) * TPB + threadIdx.x] = key[2];
		B[(i * 4 + 3) * TPB + threadIdx.x] = key[3];
	}

#pragma unroll
	for (int i = 0; i < 16; i++) input[i] = c_data[i];
#pragma unroll
	for (int i = 0; i < 8; i++) key[i] = B[i * TPB + threadIdx.x];
#pragma unroll
	for (int i = 8; i < 16; i++) key[i] = 0;

	for (int i = 0; i < 31; i++)
	{
		Blake2S(input, input, key);

		bufidx = 0;
#pragma unroll
		for (int x = 0; x < 8; ++x)
			bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
		bufidx += bufidx >> 16;
		bufidx &= 0x000000ff;
		qbuf = bufidx >> 2;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;

		shift = 32U - bitbuf;
		temp[0] = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] ^ (input[0] << bitbuf);
		temp[1] = B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[0], input[1], shift);
		temp[2] = B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[1], input[2], shift);
		temp[3] = B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[2], input[3], shift);
		temp[4] = B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[3], input[4], shift);
		temp[5] = B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[4], input[5], shift);
		temp[6] = B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[5], input[6], shift);
		temp[7] = B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] ^ __funnelshift_rc(input[6], input[7], shift);
		temp[8] = B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] ^ (input[7] >> shift);

		B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] = temp[0];
		B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] = temp[1];
		B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] = temp[2];
		B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] = temp[3];
		B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] = temp[4];
		B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] = temp[5];
		B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] = temp[6];
		B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] = temp[7];
		B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] = temp[8];

		noncepos = qbuf < 20U ? 19U : 39U;
		noncepos = qbuf < 40U ? noncepos : 59U;

		a = (qbuf + 0) & 0x3f;
		a = a == noncepos ? nonce : c_data[a];

#pragma unroll
		for (int k = 0; k < 16; k += 2)
		{
			b = (qbuf + k + 1) & 0x3f;
			b = b == noncepos ? nonce : c_data[b];
			input[k + 0] = __funnelshift_rc(a, b, bitbuf);

			a = (qbuf + k + 2) & 0x3f;
			a = a == noncepos ? nonce : c_data[a];
			input[k + 1] = __funnelshift_rc(b, a, bitbuf);
		}

		key[0] = __funnelshift_rc(temp[0], temp[1], bitbuf);
		key[1] = __funnelshift_rc(temp[1], temp[2], bitbuf);
		key[2] = __funnelshift_rc(temp[2], temp[3], bitbuf);
		key[3] = __funnelshift_rc(temp[3], temp[4], bitbuf);
		key[4] = __funnelshift_rc(temp[4], temp[5], bitbuf);
		key[5] = __funnelshift_rc(temp[5], temp[6], bitbuf);
		key[6] = __funnelshift_rc(temp[6], temp[7], bitbuf);
		key[7] = __funnelshift_rc(temp[7], temp[8], bitbuf);
	}

	Blake2S(input, input, key);

	bufidx = 0;
#pragma unroll
	for (int x = 0; x < 8; ++x)
		bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
	bufidx += bufidx >> 16;
	bufidx &= 0x000000ff;
	qbuf = bufidx >> 2;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	temp[7] = B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x];
	temp[8] = B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x];

	uint32_t output = __funnelshift_rc(temp[7], temp[8], bitbuf) ^ input[7] ^ c_data[7];

	return output;
}

#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA_host[idx0][idx1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,16); \
	c += d; b = ROTR32(b^c, 12); \
	idx = BLAKE2S_SIGMA_host[idx0][(idx1)+1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,8); \
	c += d; b = ROTR32(b^c, 7); \
}

static void Blake2Shost(uint32_t * inout, const uint32_t * inkey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}

__global__
__launch_bounds__(TPB, BPM2)
void neoscrypt_gpu_hash_start(const uint32_t threads, const int stratum, const uint32_t startNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;
	const uint32_t ZNonce = (stratum) ? cuda_swab32(nonce) : nonce; //freaking morons !!!
	if (thread < threads) {

		fastkdf256(thread, ZNonce);
	}
}

__global__
__launch_bounds__(TPB, BPM1)
void neoscrypt_gpu_hash_chacha1(const uint32_t threads, const uint32_t start)
{
#if __CUDA_ARCH__ >= 500
	const uint32_t thread = blockIdx.x * TPB + threadIdx.x;

	if (thread < threads) {
		uint4 X[16];
#pragma unroll
		for (int i = 0; i < 16; i++) {
			X[i] = __ldg(Input + 16U * (thread + start) + i);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 0, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 1, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 2, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 3, X[3]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4, X[4]);
			__stL1(W + (SHIFT * thread + i) * 16U + 5, X[5]);
			__stL1(W + (SHIFT * thread + i) * 16U + 6, X[6]);
			__stL1(W + (SHIFT * thread + i) * 16U + 7, X[7]);
			__stL1(W + (SHIFT * thread + i) * 16U + 8, X[8]);
			__stL1(W + (SHIFT * thread + i) * 16U + 9, X[9]);
			__stL1(W + (SHIFT * thread + i) * 16U + 10, X[10]);
			__stL1(W + (SHIFT * thread + i) * 16U + 11, X[11]);
			__stL1(W + (SHIFT * thread + i) * 16U + 12, X[12]);
			__stL1(W + (SHIFT * thread + i) * 16U + 13, X[13]);
			__stL1(W + (SHIFT * thread + i) * 16U + 14, X[14]);
			__stL1(W + (SHIFT * thread + i) * 16U + 15, X[15]);

			neoscrypt_chacha(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = X[12].x & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 0);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 1);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 2);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 3);
			X[4] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4);
			X[5] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 5);
			X[6] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 6);
			X[7] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 7);
			X[8] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 8);
			X[9] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 9);
			X[10] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 10);
			X[11] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 11);
			X[12] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 12);
			X[13] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 13);
			X[14] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 14);
			X[15] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 15);

			neoscrypt_chacha(X);
		}

#pragma unroll
		for (uint32_t i = 0; i < 16; i++)
		{
			Tr[16U * (thread + start) + i] = X[i];
		}
	}
#else
	const uint32_t thread = blockIdx.x * (TPB >> 2) + threadIdx.y;

	if (thread < threads) {
		uint4 X[4];
#pragma unroll
		for (uint32_t i = 0; i < 4; i++)
		{
			X[i].x = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 0 * 4 + threadIdx.x);
			X[i].y = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 1 * 4 + threadIdx.x);
			X[i].z = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 2 * 4 + threadIdx.x);
			X[i].w = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 3 * 4 + threadIdx.x);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 0 + threadIdx.x, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 1 + threadIdx.x, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 2 + threadIdx.x, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 3 + threadIdx.x, X[3]);

			neoscrypt_chacha_parallel(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = WarpShuffle(X[3].x, 0, 4) & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 0 + threadIdx.x);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 1 + threadIdx.x);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 2 + threadIdx.x);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 3 + threadIdx.x);

			neoscrypt_chacha_parallel(X);
		}

#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 0 * 4 + threadIdx.x] = X[i].x;
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 1 * 4 + threadIdx.x] = X[i].y;
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 2 * 4 + threadIdx.x] = X[i].z;
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 3 * 4 + threadIdx.x] = X[i].w;
		}
	}
#endif
}

__global__
__launch_bounds__(TPB, BPM1)
void neoscrypt_gpu_hash_salsa1(const uint32_t threads, const uint32_t start)
{
#if __CUDA_ARCH__ >= 500
	const uint32_t thread = blockIdx.x * TPB + threadIdx.x;

	if (thread < threads) {
		uint4 X[16];
#pragma unroll
		for (uint32_t i = 0; i < 16; i++)
		{
			X[i] = __ldg(Input + 16U * (thread + start) + i);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 0, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 1, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 2, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 3, X[3]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4, X[4]);
			__stL1(W + (SHIFT * thread + i) * 16U + 5, X[5]);
			__stL1(W + (SHIFT * thread + i) * 16U + 6, X[6]);
			__stL1(W + (SHIFT * thread + i) * 16U + 7, X[7]);
			__stL1(W + (SHIFT * thread + i) * 16U + 8, X[8]);
			__stL1(W + (SHIFT * thread + i) * 16U + 9, X[9]);
			__stL1(W + (SHIFT * thread + i) * 16U + 10, X[10]);
			__stL1(W + (SHIFT * thread + i) * 16U + 11, X[11]);
			__stL1(W + (SHIFT * thread + i) * 16U + 12, X[12]);
			__stL1(W + (SHIFT * thread + i) * 16U + 13, X[13]);
			__stL1(W + (SHIFT * thread + i) * 16U + 14, X[14]);
			__stL1(W + (SHIFT * thread + i) * 16U + 15, X[15]);

			neoscrypt_salsa(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = X[12].x & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 0);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 1);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 2);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 3);
			X[4] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4);
			X[5] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 5);
			X[6] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 6);
			X[7] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 7);
			X[8] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 8);
			X[9] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 9);
			X[10] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 10);
			X[11] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 11);
			X[12] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 12);
			X[13] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 13);
			X[14] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 14);
			X[15] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 15);

			neoscrypt_salsa(X);
		}

#pragma unroll
		for (uint32_t i = 0; i < 16; i++)
		{
			Tr2[16U * (thread + start) + i] = X[i];
		}
	}
#else
	const uint32_t thread = blockIdx.x * (TPB >> 2) + threadIdx.y;

	if (thread < threads) {
		uint4 X[4];
#pragma unroll
		for (uint32_t i = 0; i < 4; i++)
		{
			X[i].x = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((0 + threadIdx.x) & 3) * 4 + threadIdx.x);
			X[i].y = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((1 + threadIdx.x) & 3) * 4 + threadIdx.x);
			X[i].z = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((2 + threadIdx.x) & 3) * 4 + threadIdx.x);
			X[i].w = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((3 + threadIdx.x) & 3) * 4 + threadIdx.x);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 0 + threadIdx.x, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 1 + threadIdx.x, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 2 + threadIdx.x, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 3 + threadIdx.x, X[3]);

			neoscrypt_salsa_parallel(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = WarpShuffle(X[3].x, 0, 4) & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 0 + threadIdx.x);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 1 + threadIdx.x);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 2 + threadIdx.x);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 3 + threadIdx.x);

			neoscrypt_salsa_parallel(X);
		}

#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((0 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].x;
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((1 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].y;
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((2 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].z;
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((3 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].w;
		}
	}
#endif
}

__global__
__launch_bounds__(TPB, BPM2)
void neoscrypt_gpu_hash_ending(const uint32_t threads, const int stratum, const uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;
	const uint32_t ZNonce = (stratum) ? cuda_swab32(nonce) : nonce;
	if (thread < threads) {
		uint32_t outbuf = fastkdf32(thread, ZNonce);

		if (outbuf <= c_target[1])
		{
			uint32_t tmp = atomicExch(resNonces, nonce);
			if (tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

#define SPLIT_COUNT 1

static THREAD uint32_t *hash1 = NULL;
static THREAD uint32_t *Trans1 = NULL;
static THREAD uint32_t *Trans2 = NULL; // 2 streams
static THREAD uint32_t *Trans3 = NULL; // 2 streams

__host__
void neoscrypt_init(int thr_id, uint32_t threads)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb;
	if (device_sm[dev_id] >= 700) tpb = TPB60;
	else if (device_sm[dev_id] >= 610) tpb = TPB52;
	else if (device_sm[dev_id] >= 600) tpb = TPB60;
	else if (device_sm[dev_id] >= 520) tpb = TPB52;
	else if (device_sm[dev_id] >= 500) tpb = TPB50;
	else if (device_sm[dev_id] >= 300) tpb = TPB30;
	else tpb = TPB20;

	CUDA_SAFE_CALL(cudaMalloc(&d_NNonce[thr_id], 2 * sizeof(uint32_t)));

	if (device_sm[dev_id] < 500) {
		CUDA_SAFE_CALL(cudaMalloc(&hash1, 16 * 128 * sizeof(uint32_t) * ((threads * 4 / SPLIT_COUNT + tpb - 1) / tpb) * tpb));
	}
	else {
		CUDA_SAFE_CALL(cudaMalloc(&hash1, 4 * 16 * 128 * sizeof(uint32_t) * ((threads / SPLIT_COUNT + tpb - 1) / tpb) * tpb));
	}
	CUDA_SAFE_CALL(cudaMalloc(&Trans1, 64 * sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans2, 64 * sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans3, 64 * sizeof(uint32_t) * threads));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(W, &hash1, sizeof(uint4*), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(Tr, &Trans1, sizeof(uint4*), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(Tr2, &Trans2, sizeof(uint4*), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(Input, &Trans3, sizeof(uint4*), 0, cudaMemcpyHostToDevice));
}

__host__
void neoscrypt_free(int thr_id)
{
	cudaFree(d_NNonce[thr_id]);

	cudaFree(hash1);
	cudaFree(Trans1);
	cudaFree(Trans2);
	cudaFree(Trans3);
}

__host__
void neoscrypt_hash_k4(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces, bool stratum, uint32_t bpm)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb, sm1, sm2 = 0;

	if (device_sm[dev_id] >= 700) { tpb = TPB60; sm1 = TPB60 * 256; sm2 = bpm >= 8 ? 0 : 65536 / bpm; }
	else if (device_sm[dev_id] >= 610) { tpb = TPB52; sm1 = TPB52 * 256; sm2 = bpm >= 4 ? 0 : 98304 / bpm; }
	else if (device_sm[dev_id] >= 600) { tpb = TPB60; sm1 = TPB60 * 256; sm2 = bpm >= 8 ? 0 : 65536 / bpm; }
	else if (device_sm[dev_id] >= 520) { tpb = TPB52; sm1 = TPB52 * 256; sm2 = bpm >= 4 ? 0 : 98304 / bpm; }
	else if (device_sm[dev_id] >= 500) { tpb = TPB50; sm1 = TPB50 * 256; sm2 = bpm >= 4 ? 0 : 65536 / bpm; }
	else if (device_sm[dev_id] >= 300) { tpb = TPB30; sm1 = TPB30 * 256; sm2 = 0; }
	else { tpb = TPB20; sm1 = TPB20 * 256; sm2 = TPB20 * 12; }

	CUDA_SAFE_CALL(cudaMemset(d_NNonce[thr_id], 0xff, 2 * sizeof(uint32_t)));

	dim3 grid1((threads + tpb - 1) / tpb);
	dim3 block1(tpb);

	dim3 grid2((threads / SPLIT_COUNT + tpb - 1) / tpb);
	dim3 block2(tpb);

	dim3 grid3((threads * 4 / SPLIT_COUNT + tpb - 1) / tpb);
	dim3 block3(4, tpb >> 2);

	if (device_sm[dev_id] < 500) {
		cudaFuncSetCacheConfig(neoscrypt_gpu_hash_start, cudaFuncCachePreferEqual);
		cudaFuncSetCacheConfig(neoscrypt_gpu_hash_ending, cudaFuncCachePreferEqual);
	}
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	else if (device_sm[dev_id] >= 700) {
		cudaFuncSetAttribute(neoscrypt_gpu_hash_start, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(neoscrypt_gpu_hash_ending, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
	}
#endif

	neoscrypt_gpu_hash_start << <grid1, block1, sm1, gpustream[thr_id] >> > (threads, stratum, startNounce); //fastkdf
	CUDA_SAFE_CALL(cudaGetLastError());

	if (device_sm[dev_id] < 500) {
		for (int i = 0; i < SPLIT_COUNT; i++) {
			uint32_t start = i * grid3.x * (tpb >> 2);
			uint32_t count = min(grid3.x * (tpb >> 2), threads - start);
			neoscrypt_gpu_hash_chacha1 << <grid3, block3, sm2, gpustream[thr_id] >> > (count, start);
			CUDA_SAFE_CALL(cudaGetLastError());
			neoscrypt_gpu_hash_salsa1 << <grid3, block3, sm2, gpustream[thr_id] >> > (count, start);
			CUDA_SAFE_CALL(cudaGetLastError());
		}
	}
	else {
		for (int i = 0; i < SPLIT_COUNT; i++) {
			uint32_t start = i * grid2.x * tpb;
			uint32_t count = min(grid2.x * tpb, threads - start);
			neoscrypt_gpu_hash_chacha1 << <grid2, block2, sm2, gpustream[thr_id] >> > (count, start);
			CUDA_SAFE_CALL(cudaGetLastError());
			neoscrypt_gpu_hash_salsa1 << <grid2, block2, sm2, gpustream[thr_id] >> > (count, start);
			CUDA_SAFE_CALL(cudaGetLastError());
		}
	}
	neoscrypt_gpu_hash_ending << <grid1, block1, sm1 >> > (threads, stratum, startNounce, d_NNonce[thr_id]); //fastkdf+end
	CUDA_SAFE_CALL(cudaGetLastError());

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_NNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__host__
void neoscrypt_setBlockTarget(uint32_t thr_id, uint32_t* const pdata, uint32_t* const target)
{
	uint32_t PaddedMessage[64];
	uint32_t input[16], key[16] = { 0 };

	for (int i = 0; i < 19; i++)
	{
		PaddedMessage[i] = pdata[i];
		PaddedMessage[i + 20] = pdata[i];
		PaddedMessage[i + 40] = pdata[i];
	}
	for (int i = 0; i < 4; i++)
		PaddedMessage[i + 60] = pdata[i];

	PaddedMessage[19] = 0;
	PaddedMessage[39] = 0;
	PaddedMessage[59] = 0;

	((uint16*)input)[0] = ((uint16*)pdata)[0];
	((uint8*)key)[0] = ((uint8*)pdata)[0];

	Blake2Shost(input, key);

	cudaMemcpyToSymbol(input_init, input, 64, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(key_init, key, 64, 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_target, &target[6], 2 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, PaddedMessage, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaGetLastError());
}

