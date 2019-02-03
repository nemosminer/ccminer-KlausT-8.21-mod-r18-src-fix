/**
 * Lyra2 (v2) CUDA Implementation
 *
 * Based on djm34/VTC sources and incredible 2x boost by Nanashi Meiyo-Meijin (May 2016)
 */
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "miner.h"
#include "cuda_lyra2_vectors.h"

#ifdef __INTELLISENSE__
 /* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
__device__ void __threadfence_block();

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#if __CUDA_ARCH__ >= 300
__device__ uint32_t __shfl_sync(const uint32_t mask, uint32_t a, uint32_t b, uint32_t c);
#endif
__device__ void __syncwarp(uint32_t mask);
#elif defined(CUDART_VERSION) && CUDART_VERSION < 9000
#if __CUDA_ARCH__ >= 300
__device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
#endif
#endif
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 9000
#define __shfl_sync(mask, a, b, c) __shfl(a, b, c)
#define __syncwarp(mask) __threadfence_block()
#endif

#define TPB 32

#include "cuda_lyra2_vectors.h"

static uint32_t Mode[MAX_GPUS];

#define SHARED_4WAY_MODE 0
#define GLOBAL_16WAY_MODE 1
#define GLOBAL_4WAY_MODE 2

static uint32_t *d_gnounce[MAX_GPUS];
static uint32_t *d_GNonce[MAX_GPUS];
__constant__ uint2 c_data[10];

static uint64_t *d_matrix[MAX_GPUS];
__device__ uint2 *DMatrix;
__device__ uint2 *DState;

#if __CUDA_ARCH__ < 300
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint2 buf, result;

	__syncwarp(0xFFFFFFFF);
	buf = shared_mem[threads * 0 + thread];
	shared_mem[threads * 0 + thread] = a;
	__syncwarp(0xFFFFFFFF);
	result = shared_mem[0 * threads + bitselect(thread, b, c - 1)];
	__syncwarp(0xFFFFFFFF);
	shared_mem[threads * 0 + thread] = buf;

	return result;
}

__device__ __forceinline__ void WarpShuffle2(uint2 &d0, uint2 &d1, uint2 a0, uint2 a1, uint32_t b0, uint32_t b1, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint2 buf0, buf1;

	__syncwarp(0xFFFFFFFF);
	buf0 = shared_mem[threads * 0 + thread];
	buf1 = shared_mem[threads * 1 + thread];
	shared_mem[threads * 0 + thread] = a0;
	shared_mem[threads * 1 + thread] = a1;
	__syncwarp(0xFFFFFFFF);
	d0 = shared_mem[0 * threads + bitselect(thread, b0, c - 1)];
	d1 = shared_mem[1 * threads + bitselect(thread, b1, c - 1)];
	__syncwarp(0xFFFFFFFF);
	shared_mem[threads * 0 + thread] = buf0;
	shared_mem[threads * 1 + thread] = buf1;
}

__device__ __forceinline__ void WarpShuffle3(uint2 &d0, uint2 &d1, uint2 &d2, uint2 a0, uint2 a1, uint2 a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint2 buf0, buf1, buf2;

	__syncwarp(0xFFFFFFFF);
	buf0 = shared_mem[threads * 0 + thread];
	buf1 = shared_mem[threads * 1 + thread];
	buf2 = shared_mem[threads * 2 + thread];
	shared_mem[threads * 0 + thread] = a0;
	shared_mem[threads * 1 + thread] = a1;
	shared_mem[threads * 2 + thread] = a2;
	__syncwarp(0xFFFFFFFF);
	d0 = shared_mem[0 * threads + bitselect(thread, b0, c - 1)];
	d1 = shared_mem[1 * threads + bitselect(thread, b1, c - 1)];
	d2 = shared_mem[2 * threads + bitselect(thread, b2, c - 1)];
	__syncwarp(0xFFFFFFFF);
	shared_mem[threads * 0 + thread] = buf0;
	shared_mem[threads * 1 + thread] = buf1;
	shared_mem[threads * 2 + thread] = buf2;
}

__device__ __forceinline__ void WarpShuffle4(uint2 &d0, uint2 &d1, uint2 &d2, uint2 &d3, uint2 a0, uint2 a1, uint2 a2, uint2 a3, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint2 buf0, buf1, buf2, buf3;

	__syncwarp(0xFFFFFFFF);
	buf0 = shared_mem[threads * 0 + thread];
	buf1 = shared_mem[threads * 1 + thread];
	buf2 = shared_mem[threads * 2 + thread];
	buf3 = shared_mem[threads * 3 + thread];
	shared_mem[threads * 0 + thread] = a0;
	shared_mem[threads * 1 + thread] = a1;
	shared_mem[threads * 2 + thread] = a2;
	shared_mem[threads * 3 + thread] = a3;
	__syncwarp(0xFFFFFFFF);
	d0 = shared_mem[0 * threads + bitselect(thread, b0, c - 1)];
	d1 = shared_mem[1 * threads + bitselect(thread, b1, c - 1)];
	d2 = shared_mem[2 * threads + bitselect(thread, b2, c - 1)];
	d3 = shared_mem[3 * threads + bitselect(thread, b3, c - 1)];
	__syncwarp(0xFFFFFFFF);
	shared_mem[threads * 0 + thread] = buf0;
	shared_mem[threads * 1 + thread] = buf1;
	shared_mem[threads * 2 + thread] = buf2;
	shared_mem[threads * 3 + thread] = buf3;
}
#else
__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(SHFL(a.x, b, c), SHFL(a.y, b, c));
}

__device__ __forceinline__ void WarpShuffle2(uint2 &d0, uint2 &d1, uint2 a0, uint2 a1, uint32_t b0, uint32_t b1, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
}

__device__ __forceinline__ void WarpShuffle3(uint2 &d0, uint2 &d1, uint2 &d2, uint2 a0, uint2 a1, uint2 a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
	d2 = WarpShuffle(a2, b2, c);
}

__device__ __forceinline__ void WarpShuffle4(uint2 &d0, uint2 &d1, uint2 &d2, uint2 &d3, uint2 a0, uint2 a1, uint2 a2, uint2 a3, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
	d2 = WarpShuffle(a2, b2, c);
	d3 = WarpShuffle(a3, b3, c);
}

#endif

static __device__ __forceinline__ uint2 __ldL1(const uint2 *ptr)
{
	uint2 ret;
	asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint2 *ptr, const uint2 value)
{
	asm("st.global.wb.v2.u32 [%0], {%1, %2};" ::__LDG_PTR(ptr), "r"(value.x), "r"(value.y));
}

static __device__ __forceinline__ void prefetch(const uint2 *ptr)
{
	asm("prefetch.global.L1 [%0+0];" ::__LDG_PTR(ptr));
	asm("prefetch.global.L1 [%0+4];" ::__LDG_PTR(ptr));
}

__device__ __forceinline__
void G(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	uint32_t tmp;
	a += b; d ^= a; tmp = d.x; d.x = d.y; d.y = tmp;
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__
void round_lyra(uint2x4 s[4], const uint32_t count)
{
	G(s[0].x, s[1].x, s[2].x, s[3].x);
	G(s[0].y, s[1].y, s[2].y, s[3].y);
	G(s[0].z, s[1].z, s[2].z, s[3].z);
	G(s[0].w, s[1].w, s[2].w, s[3].w);

	G(s[0].x, s[1].y, s[2].z, s[3].w);
	G(s[0].y, s[1].z, s[2].w, s[3].x);
	G(s[0].z, s[1].w, s[2].x, s[3].y);
	G(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	G(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	G(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

__device__ __forceinline__ void round_lyra(uint2 &s)
{
	uint2 s0, s1, s2, s3;
	WarpShuffle4(s0, s1, s2, s3, s, s, s, s, threadIdx.x + 0, threadIdx.x + 4, threadIdx.x + 8, threadIdx.x + 12, 16);
	G(s0, s1, s2, s3);
	WarpShuffle3(s1, s2, s3, s1, s2, s3, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	G(s0, s1, s2, s3);
	WarpShuffle3(s1, s2, s3, s1, s2, s3, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
	if (threadIdx.y == 0) s = s0;
	else if (threadIdx.y == 1) s = s1;
	else if (threadIdx.y == 2) s = s2;
	else if (threadIdx.y == 3) s = s3;
}

__device__ __forceinline__ void blake2bLyra(uint2x4 v[4]) {
	round_lyra(v, 0);
	round_lyra(v, 1);
	round_lyra(v, 2);
	round_lyra(v, 3);
	round_lyra(v, 4);
	round_lyra(v, 5);
	round_lyra(v, 6);
	round_lyra(v, 7);
	round_lyra(v, 8);
	round_lyra(v, 9);
	round_lyra(v, 10);
	round_lyra(v, 11);
}

__device__ __forceinline__ void reducedBlake2bLyra(uint2 v[4]) {
	round_lyra(v);
}

__device__ __forceinline__ void reducedBlake2bLyra(uint2 &v) {
	round_lyra(v);
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_1(uint32_t threads, const uint32_t startNonce, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint2 *Hash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	const uint2x4 blake2b_IV[2] = {
		0xf3bcc908UL, 0x6a09e667UL, 0x84caa73bUL, 0xbb67ae85UL,
		0xfe94f82bUL, 0x3c6ef372UL, 0x5f1d36f1UL, 0xa54ff53aUL,
		0xade682d1UL, 0x510e527fUL, 0x2b3e6c1fUL, 0x9b05688cUL,
		0xfb41bd6bUL, 0x1f83d9abUL, 0x137e2179UL, 0x5be0cd19UL
	};

	const uint2x4 Mask[2] = {
		0x00000020UL, 0x00000000UL, 0x00000020UL, 0x00000000UL,
		0x00000020UL, 0x00000000UL, timeCost    , 0x00000000UL,
		nRows       , 0x00000000UL, nCols       , 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0].x = state[1].x = Hash[thread + threads * 0];
		state[0].y = state[1].y = Hash[thread + threads * 1];
		state[0].z = state[1].z = Hash[thread + threads * 2];
		state[0].w = state[1].w = Hash[thread + threads * 3];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		blake2bLyra(state);

		if (!bug) {
			state[0] ^= Mask[0];
			state[1] ^= Mask[1];
		}

		blake2bLyra(state);

		((uint2x4*)DState)[thread * 4 + 0] = state[0];
		((uint2x4*)DState)[thread * 4 + 1] = state[1];
		((uint2x4*)DState)[thread * 4 + 2] = state[2];
		((uint2x4*)DState)[thread * 4 + 3] = state[3];
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_80_1(uint32_t threads, const uint32_t startNonce, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	const uint2x4 blake2b_IV[2] = {
		0xf3bcc908UL, 0x6a09e667UL, 0x84caa73bUL, 0xbb67ae85UL,
		0xfe94f82bUL, 0x3c6ef372UL, 0x5f1d36f1UL, 0xa54ff53aUL,
		0xade682d1UL, 0x510e527fUL, 0x2b3e6c1fUL, 0x9b05688cUL,
		0xfb41bd6bUL, 0x1f83d9abUL, 0x137e2179UL, 0x5be0cd19UL
	};

	const uint2x4 Mask[3] = {
		0x00000020UL, 0x00000000UL, 0x00000050UL, 0x00000000UL,
		0x00000050UL, 0x00000000UL, timeCost    , 0x00000000UL,
		nRows       , 0x00000000UL, nCols       , 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x00000000UL,
		0x00000000UL, 0x00000000UL, 0x00000000UL, 0x00000000UL,
		0x00000000UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0].x = c_data[0];
		state[0].y = c_data[1];
		state[0].z = c_data[2];
		state[0].w = c_data[3];
		state[1].x = c_data[4];
		state[1].y = c_data[5];
		state[1].z = c_data[6];
		state[1].w = c_data[7];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		blake2bLyra(state);

		if (!bug) {
			state[0].x ^= c_data[8];
			state[0].y.x ^= c_data[9].x;
			state[0].y.y ^= cuda_swab32(startNonce + thread);
			state[0].z ^= c_data[0];
			state[0].w ^= c_data[1];
			state[1].x ^= c_data[2];
			state[1].y ^= c_data[3];
			state[1].z ^= c_data[4];
			state[1].w ^= c_data[5];
		}

		blake2bLyra(state);

		if (!bug) {
			state[0].x ^= c_data[6];
			state[0].y ^= c_data[7];
			state[0].z ^= c_data[8];
			state[0].w.x ^= c_data[9].x;
			state[0].w.y ^= cuda_swab32(startNonce + thread);
			state[1] ^= Mask[0];
		}

		blake2bLyra(state);

		if (!bug) {
			state[0] ^= Mask[1];
			state[1] ^= Mask[2];
		}

		blake2bLyra(state);

		((uint2x4*)DState)[thread * 4 + 0] = state[0];
		((uint2x4*)DState)[thread * 4 + 1] = state[1];
		((uint2x4*)DState)[thread * 4 + 2] = state[2];
		((uint2x4*)DState)[thread * 4 + 3] = state[3];
	}
}

#define Mdev(r,c) DMatrix[(((blockIdx.x * nRows + (r)) * nCols + (c)) * 2 + threadIdx.z) * 12 + threadIdx.y * 4 + threadIdx.x]
#define Sdev(r,c) shared_mem[(((r) * nCols + (c)) * 2 + threadIdx.z) * 12 + threadIdx.y * 4 + threadIdx.x]
#define Ddev(r,c,a) DMatrix[((((blockIdx.x * nRows + (r)) * nCols + (c)) * 3 + (a)) * 8 + threadIdx.y) * 4 + threadIdx.x]
#define Fdev(r,c,a) shared_mem[((((r) * nCols + (c)) * 3 + (a)) * 8 + threadIdx.y) * 4 + threadIdx.x]

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_2_16way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols)
{
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;
	uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

	if (subthread == 0) subthread = 11;
	else subthread -= 1;

	if ((thread + offset) < threads)
	{
		uint32_t row, prev, rowa, step, window, gap;
		uint2 bufIn, bufInOut, bufOut, bufstate;

		uint2 state = __ldg(&DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x]);

		for (uint32_t i = 0; i < nCols; i++) {
			if (threadIdx.y < 3) __stL1(&Mdev(0, nCols - i - 1), state);

			reducedBlake2bLyra(state);
		}

		for (uint32_t i = 0; i < nCols; i++) {
			if (threadIdx.y < 3) {
				bufIn = __ldL1(&Mdev(0, i));
				if (i < nCols - 1) {
					prefetch(&Mdev(0, i + 1));
				}
				state ^= bufIn;
			}

			reducedBlake2bLyra(state);

			if (threadIdx.y < 3) __stL1(&Mdev(1, nCols - i - 1), bufIn ^ state);
		}

		for (row = 2, prev = 1, rowa = 0, step = 1, window = 2, gap = 1; row < nRows; row++) {

			for (uint32_t i = 0; i < nCols; i++) {
				if (threadIdx.y < 3) {
					bufIn = __ldL1(&Mdev(prev, i));
					bufInOut = __ldL1(&Mdev(rowa, i));
					if (i < nCols - 1) {
						prefetch(&Mdev(prev, i + 1));
						prefetch(&Mdev(rowa, i + 1));
					}
					state ^= bufIn + bufInOut;
				}

				reducedBlake2bLyra(state);

				bufstate = WarpShuffle(state, subthread, 16);

				if (threadIdx.y < 3) {
					bufOut = bufIn ^ state;
					__stL1(&Mdev(row, nCols - i - 1), bufOut);
					__stL1(&Mdev(rowa, i), bufInOut ^ bufstate);
				}
			}

			rowa = (rowa + step) & (window - 1);
			prev = row;

			if (rowa == 0) {
				step = window + gap;
				window <<= 1;
				gap = -gap;
			}
		}
		DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_2_4way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols)
{
	const uint32_t thread = blockIdx.x * blockDim.y + threadIdx.y;

	if ((thread + offset) < threads)
	{
		uint32_t row, prev, rowa, step, window, gap;
		uint2 bufIn[3], bufInOut[3], bufOut[3], bufstate[3];

		uint2 state[4];
		state[0] = __ldg(&DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x]);
		state[1] = __ldg(&DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x]);
		state[2] = __ldg(&DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x]);
		state[3] = __ldg(&DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x]);

		for (uint32_t i = 0; i < nCols; i++) {
			__stL1(&Ddev(0, nCols - i - 1, 0), state[0]);
			__stL1(&Ddev(0, nCols - i - 1, 1), state[1]);
			__stL1(&Ddev(0, nCols - i - 1, 2), state[2]);

			reducedBlake2bLyra(state);
		}

		for (uint32_t i = 0; i < nCols; i++) {
			bufIn[0] = __ldL1(&Ddev(0, i, 0));
			bufIn[1] = __ldL1(&Ddev(0, i, 1));
			bufIn[2] = __ldL1(&Ddev(0, i, 2));
			if (i < nCols - 1) {
				prefetch(&Ddev(0, i + 1, 0));
				prefetch(&Ddev(0, i + 1, 1));
				prefetch(&Ddev(0, i + 1, 2));
			}
			state[0] ^= bufIn[0];
			state[1] ^= bufIn[1];
			state[2] ^= bufIn[2];

			reducedBlake2bLyra(state);

			__stL1(&Ddev(1, nCols - i - 1, 0), bufIn[0] ^ state[0]);
			__stL1(&Ddev(1, nCols - i - 1, 1), bufIn[1] ^ state[1]);
			__stL1(&Ddev(1, nCols - i - 1, 2), bufIn[2] ^ state[2]);
		}

		for (row = 2, prev = 1, rowa = 0, step = 1, window = 2, gap = 1; row < nRows; row++) {

			for (uint32_t i = 0; i < nCols; i++) {
				bufIn[0] = __ldL1(&Ddev(prev, i, 0));
				bufIn[1] = __ldL1(&Ddev(prev, i, 1));
				bufIn[2] = __ldL1(&Ddev(prev, i, 2));
				bufInOut[0] = __ldL1(&Ddev(rowa, i, 0));
				bufInOut[1] = __ldL1(&Ddev(rowa, i, 1));
				bufInOut[2] = __ldL1(&Ddev(rowa, i, 2));
				if (i < nCols - 1) {
					prefetch(&Ddev(prev, i + 1, 0));
					prefetch(&Ddev(prev, i + 1, 1));
					prefetch(&Ddev(prev, i + 1, 2));
					prefetch(&Ddev(rowa, i + 1, 0));
					prefetch(&Ddev(rowa, i + 1, 1));
					prefetch(&Ddev(rowa, i + 1, 2));
				}
				state[0] ^= bufIn[0] + bufInOut[0];
				state[1] ^= bufIn[1] + bufInOut[1];
				state[2] ^= bufIn[2] + bufInOut[2];

				reducedBlake2bLyra(state);

				WarpShuffle3(bufstate[0], bufstate[1], bufstate[2], state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4);

				bufOut[0] = bufIn[0] ^ state[0];
				bufOut[1] = bufIn[1] ^ state[1];
				bufOut[2] = bufIn[2] ^ state[2];
				__stL1(&Ddev(row, nCols - i - 1, 0), bufOut[0]);
				__stL1(&Ddev(row, nCols - i - 1, 1), bufOut[1]);
				__stL1(&Ddev(row, nCols - i - 1, 2), bufOut[2]);

				if (threadIdx.x == 0) {
					bufInOut[0] ^= bufstate[2];
					bufInOut[1] ^= bufstate[0];
					bufInOut[2] ^= bufstate[1];
				}
				else {
					bufInOut[0] ^= bufstate[0];
					bufInOut[1] ^= bufstate[1];
					bufInOut[2] ^= bufstate[2];
				}

				__stL1(&Ddev(rowa, i, 0), bufInOut[0]);
				__stL1(&Ddev(rowa, i, 1), bufInOut[1]);
				__stL1(&Ddev(rowa, i, 2), bufInOut[2]);
			}

			rowa = (rowa + step) & (window - 1);
			prev = row;

			if (rowa == 0) {
				step = window + gap;
				window <<= 1;
				gap = -gap;
			}
		}
		DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x] = state[0];
		DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x] = state[1];
		DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x] = state[2];
		DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_2_shared_4way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols)
{
	extern __shared__ uint2 shared_mem[];
	const uint32_t thread = blockIdx.x * blockDim.y + threadIdx.y;

	if ((thread + offset) < threads)
	{
		uint32_t row, prev, rowa, step, window, gap;
		uint2 bufIn[3], bufInOut[3], bufOut[3], bufstate[3];

		uint2 state[4];
		state[0] = __ldg(&DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x]);
		state[1] = __ldg(&DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x]);
		state[2] = __ldg(&DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x]);
		state[3] = __ldg(&DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x]);

		for (uint32_t i = 0; i < nCols; i++) {
			Fdev(0, nCols - i - 1, 0) = state[0];
			Fdev(0, nCols - i - 1, 1) = state[1];
			Fdev(0, nCols - i - 1, 2) = state[2];

			reducedBlake2bLyra(state);
		}

		for (uint32_t i = 0; i < nCols; i++) {
			bufIn[0] = Fdev(0, i, 0);
			bufIn[1] = Fdev(0, i, 1);
			bufIn[2] = Fdev(0, i, 2);
			state[0] ^= bufIn[0];
			state[1] ^= bufIn[1];
			state[2] ^= bufIn[2];

			reducedBlake2bLyra(state);

			Fdev(1, nCols - i - 1, 0) = bufIn[0] ^ state[0];
			Fdev(1, nCols - i - 1, 1) = bufIn[1] ^ state[1];
			Fdev(1, nCols - i - 1, 2) = bufIn[2] ^ state[2];
		}

		for (row = 2, prev = 1, rowa = 0, step = 1, window = 2, gap = 1; row < nRows; row++) {

			for (uint32_t i = 0; i < nCols; i++) {
				bufIn[0] = Fdev(prev, i, 0);
				bufIn[1] = Fdev(prev, i, 1);
				bufIn[2] = Fdev(prev, i, 2);
				bufInOut[0] = Fdev(rowa, i, 0);
				bufInOut[1] = Fdev(rowa, i, 1);
				bufInOut[2] = Fdev(rowa, i, 2);
				state[0] ^= bufIn[0] + bufInOut[0];
				state[1] ^= bufIn[1] + bufInOut[1];
				state[2] ^= bufIn[2] + bufInOut[2];

				reducedBlake2bLyra(state);

				WarpShuffle3(bufstate[0], bufstate[1], bufstate[2], state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4);

				bufOut[0] = bufIn[0] ^ state[0];
				bufOut[1] = bufIn[1] ^ state[1];
				bufOut[2] = bufIn[2] ^ state[2];

				Fdev(row, nCols - i - 1, 0) = bufOut[0];
				Fdev(row, nCols - i - 1, 1) = bufOut[1];
				Fdev(row, nCols - i - 1, 2) = bufOut[2];

				if (threadIdx.x == 0) {
					bufInOut[0] ^= bufstate[2];
					bufInOut[1] ^= bufstate[0];
					bufInOut[2] ^= bufstate[1];
				}
				else {
					bufInOut[0] ^= bufstate[0];
					bufInOut[1] ^= bufstate[1];
					bufInOut[2] ^= bufstate[2];
				}

				Fdev(rowa, i, 0) = bufInOut[0];
				Fdev(rowa, i, 1) = bufInOut[1];
				Fdev(rowa, i, 2) = bufInOut[2];
			}

			rowa = (rowa + step) & (window - 1);
			prev = row;

			if (rowa == 0) {
				step = window + gap;
				window <<= 1;
				gap = -gap;
			}
		}

		row = 0;
		for (uint32_t tau = 1; tau <= timeCost; tau++)
		{
			uint64_t step = (tau & 1) == 0 ? 0xFFFFFFFFFFFFFFFF : (uint64_t)((nRows >> 1) - 1);

			do {
				rowa = (uint32_t)(devectorize(WarpShuffle(state[0], 0, 4)) % (uint64_t)nRows);

				for (uint32_t i = 0; i < nCols; i++) {
					bufIn[0] = Fdev(prev, i, 0);
					bufIn[1] = Fdev(prev, i, 1);
					bufIn[2] = Fdev(prev, i, 2);
					bufInOut[0] = Fdev(rowa, i, 0);
					bufInOut[1] = Fdev(rowa, i, 1);
					bufInOut[2] = Fdev(rowa, i, 2);
					bufOut[0] = Fdev(row, i, 0);
					bufOut[1] = Fdev(row, i, 1);
					bufOut[2] = Fdev(row, i, 2);
					state[0] ^= bufIn[0] + bufInOut[0];
					state[1] ^= bufIn[1] + bufInOut[1];
					state[2] ^= bufIn[2] + bufInOut[2];

					reducedBlake2bLyra(state);

					WarpShuffle3(bufstate[0], bufstate[1], bufstate[2], state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4);

					bufOut[0] ^= state[0];
					bufOut[1] ^= state[1];
					bufOut[2] ^= state[2];
					if (row == rowa) {
						bufInOut[0] = bufOut[0];
						bufInOut[1] = bufOut[1];
						bufInOut[2] = bufOut[2];
					}
					else {
						Fdev(row, i, 0) = bufOut[0];
						Fdev(row, i, 1) = bufOut[1];
						Fdev(row, i, 2) = bufOut[2];
					}

					if (threadIdx.x == 0) {
						bufInOut[0] ^= bufstate[2];
						bufInOut[1] ^= bufstate[0];
						bufInOut[2] ^= bufstate[1];
					}
					else {
						bufInOut[0] ^= bufstate[0];
						bufInOut[1] ^= bufstate[1];
						bufInOut[2] ^= bufstate[2];
					}
					Fdev(rowa, i, 0) = bufInOut[0];
					Fdev(rowa, i, 1) = bufInOut[1];
					Fdev(rowa, i, 2) = bufInOut[2];
				}

				prev = row;

				row = (uint32_t)(((uint64_t)row + step) % (uint64_t)nRows);
			} while (row != 0);
		}

		state[0] ^= Fdev(rowa, 0, 0);
		state[1] ^= Fdev(rowa, 0, 1);
		state[2] ^= Fdev(rowa, 0, 2);

		DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x] = state[0];
		DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x] = state[1];
		DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x] = state[2];
		DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_3_16way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, uint32_t tau, uint32_t prev)
{
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;

	if ((thread + offset) < threads)
	{
		uint2 state = __ldg(&DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x]);
		uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

		if (subthread == 0) subthread = 11;
		else subthread -= 1;

		uint32_t row = 0;
		uint32_t rowa;
		uint2 bufIn, bufInOut, bufOut, bufstate;
		uint64_t step = (tau & 1) == 0 ? 0xFFFFFFFFFFFFFFFF : (uint64_t)((nRows >> 1) - 1);

		do {
			rowa = (uint32_t)(devectorize(WarpShuffle(state, 0, 16)) % (uint64_t)nRows);

			for (uint32_t i = 0; i < nCols; i++) {
				if (threadIdx.y < 3) {
					bufIn = __ldL1(&Mdev(prev, i));
					bufInOut = __ldL1(&Mdev(rowa, i));
					bufOut = __ldL1(&Mdev(row, i));
					if (i < nCols - 1) {
						prefetch(&Mdev(prev, i + 1));
						prefetch(&Mdev(rowa, i + 1));
						prefetch(&Mdev(row, i + 1));
					}
					state ^= bufIn + bufInOut;
				}

				reducedBlake2bLyra(state);

				bufstate = WarpShuffle(state, subthread, 16);

				if (threadIdx.y < 3) {
					bufOut ^= state;
					if (row == rowa) bufInOut = bufOut;
					else __stL1(&Mdev(row, i), bufOut);
					__stL1(&Mdev(rowa, i), bufInOut ^ bufstate);
				}
			}

			prev = row;

			row = (uint32_t)(((uint64_t)row + step) % (uint64_t)nRows);
		} while (row != 0);

		if (tau == timeCost && threadIdx.y < 3) state ^= __ldL1(&Mdev(rowa, 0));

		DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_3_4way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, uint32_t tau, uint32_t prev)
{
	const uint32_t thread = blockIdx.x * blockDim.y + threadIdx.y;

	if ((thread + offset) < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x]);
		state[1] = __ldg(&DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x]);
		state[2] = __ldg(&DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x]);
		state[3] = __ldg(&DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x]);

		uint32_t row = 0;
		uint32_t rowa;
		uint2 bufIn[3], bufInOut[3], bufOut[3], bufstate[3];
		uint64_t step = (tau & 1) == 0 ? 0xFFFFFFFFFFFFFFFF : (uint64_t)((nRows >> 1) - 1);

		do {
			rowa = (uint32_t)(devectorize(WarpShuffle(state[0], 0, 4)) % (uint64_t)nRows);

			for (uint32_t i = 0; i < nCols; i++) {
				bufIn[0] = __ldL1(&Ddev(prev, i, 0));
				bufIn[1] = __ldL1(&Ddev(prev, i, 1));
				bufIn[2] = __ldL1(&Ddev(prev, i, 2));
				bufInOut[0] = __ldL1(&Ddev(rowa, i, 0));
				bufInOut[1] = __ldL1(&Ddev(rowa, i, 1));
				bufInOut[2] = __ldL1(&Ddev(rowa, i, 2));
				bufOut[0] = __ldL1(&Ddev(row, i, 0));
				bufOut[1] = __ldL1(&Ddev(row, i, 1));
				bufOut[2] = __ldL1(&Ddev(row, i, 2));
				if (i < nCols - 1) {
					prefetch(&Ddev(prev, i + 1, 0));
					prefetch(&Ddev(prev, i + 1, 1));
					prefetch(&Ddev(prev, i + 1, 2));
					prefetch(&Ddev(rowa, i + 1, 0));
					prefetch(&Ddev(rowa, i + 1, 1));
					prefetch(&Ddev(rowa, i + 1, 2));
					prefetch(&Ddev(row, i + 1, 0));
					prefetch(&Ddev(row, i + 1, 1));
					prefetch(&Ddev(row, i + 1, 2));
				}
				state[0] ^= bufIn[0] + bufInOut[0];
				state[1] ^= bufIn[1] + bufInOut[1];
				state[2] ^= bufIn[2] + bufInOut[2];

				reducedBlake2bLyra(state);

				WarpShuffle3(bufstate[0], bufstate[1], bufstate[2], state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4);

				bufOut[0] ^= state[0];
				bufOut[1] ^= state[1];
				bufOut[2] ^= state[2];
				if (row == rowa) {
					bufInOut[0] = bufOut[0];
					bufInOut[1] = bufOut[1];
					bufInOut[2] = bufOut[2];
				}
				else {
					__stL1(&Ddev(row, i, 0), bufOut[0]);
					__stL1(&Ddev(row, i, 1), bufOut[1]);
					__stL1(&Ddev(row, i, 2), bufOut[2]);
				}

				if (threadIdx.x == 0) {
					bufInOut[0] ^= bufstate[2];
					bufInOut[1] ^= bufstate[0];
					bufInOut[2] ^= bufstate[1];
				}
				else {
					bufInOut[0] ^= bufstate[0];
					bufInOut[1] ^= bufstate[1];
					bufInOut[2] ^= bufstate[2];
				}

				__stL1(&Ddev(rowa, i, 0), bufInOut[0]);
				__stL1(&Ddev(rowa, i, 1), bufInOut[1]);
				__stL1(&Ddev(rowa, i, 2), bufInOut[2]);
			}

			prev = row;

			row = (uint32_t)(((uint64_t)row + step) % (uint64_t)nRows);
		} while (row != 0);

		if (tau == timeCost) {
			state[0] ^= __ldL1(&Ddev(rowa, 0, 0));
			state[1] ^= __ldL1(&Ddev(rowa, 0, 1));
			state[2] ^= __ldL1(&Ddev(rowa, 0, 2));
		}

		DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x] = state[0];
		DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x] = state[1];
		DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x] = state[2];
		DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_4(uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *const __restrict__ nonceVector, uint2 *const __restrict__ Hash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DState)[thread * 4 + 0]);
		state[1] = __ldg4(&((uint2x4*)DState)[thread * 4 + 1]);
		state[2] = __ldg4(&((uint2x4*)DState)[thread * 4 + 2]);
		state[3] = __ldg4(&((uint2x4*)DState)[thread * 4 + 3]);

		blake2bLyra(state);

		if (nonceVector)
		{
			if (state[0].w.y <= target)
			{
				uint32_t tmp = atomicExch(&nonceVector[0], startNounce + thread);
				if (tmp != 0)
					nonceVector[1] = tmp;
			}
		}
		else {
			Hash[thread + threads * 0] = state[0].x;
			Hash[thread + threads * 1] = state[0].y;
			Hash[thread + threads * 2] = state[0].z;
			Hash[thread + threads * 3] = state[0].w;
		}
	}
}

__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, const uint32_t nRows, const uint32_t nCols, uint64_t *d_state)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	Mode[thr_id] = SHARED_4WAY_MODE;
	if ((device_sm[dev_id] > 500 && nRows*nCols > 64) || (device_sm[dev_id] == 500 && nRows*nCols > 42)) Mode[thr_id] = GLOBAL_16WAY_MODE;
	else if (device_sm[dev_id] < 500) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);
		if ((props.totalGlobalMem - 256 * 1024 * 1024) / (20 * sizeof(uint64_t) + 12 * nRows * nCols * sizeof(uint64_t) / 4) > threads)
			Mode[thr_id] = GLOBAL_4WAY_MODE;
		else Mode[thr_id] = GLOBAL_16WAY_MODE;
	}

	switch (Mode[thr_id])
	{
	case SHARED_4WAY_MODE:
		applog(LOG_WARNING, "Using 4-Way Mode On Shared Memory");
		break;
	case GLOBAL_4WAY_MODE:
		applog(LOG_WARNING, "Using 4-Way Mode On Global Memory");
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], sizeof(uint64_t) * 12 * nRows * nCols * ((threads + 31) / 32) * 8));
		cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
		break;
	case GLOBAL_16WAY_MODE:
		applog(LOG_WARNING, "Using 16-Way Mode On Global Memory");
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], sizeof(uint64_t) * 12 * nRows * nCols * ((threads + 31) / 32) * 2));
		cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
		break;
	}

	cudaMemcpyToSymbol(DState, &d_state, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);

	cudaMalloc(&d_GNonce[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&d_gnounce[thr_id], 2 * sizeof(uint32_t));
}

__host__
void lyra2_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint64_t *Hash)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	const uint32_t tpb = TPB;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block1(tpb);
	dim3 block2(4, 4, tpb / 16);
	dim3 block3(4, tpb / 4);
	uint32_t sm = 0;
	if (device_sm[dev_id] < 300) sm = 4 * sizeof(uint2)*tpb;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	if (device_sm[dev_id] == 700)
		cudaFuncSetAttribute(lyra2_gpu_hash_32_2_shared_4way, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
#endif

	lyra2_gpu_hash_80_1 << < grid, block1 >> > (threads, startNounce, timeCost, nRows, nCols, bug);
	switch (Mode[thr_id])
	{
	case SHARED_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			lyra2_gpu_hash_32_2_shared_4way << < grid, block3, nRows * nCols * sizeof(uint64_t) * 12 * (tpb / 4) >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
		}
		break;
	case GLOBAL_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	case GLOBAL_16WAY_MODE:
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	}
	lyra2_gpu_hash_32_4 << < grid, block1 >> > (threads, startNounce, 0, NULL, (uint2*)Hash);
}

__host__
void lyra2_cpu_hash_80_ending(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint32_t *resultnonces)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	const uint32_t tpb = TPB;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block1(tpb);
	dim3 block2(4, 4, tpb / 16);
	dim3 block3(4, tpb / 4);
	uint32_t sm = 0;
	if (device_sm[dev_id] < 300) sm = 4 * sizeof(uint2)*tpb;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	if (device_sm[dev_id] == 700)
		cudaFuncSetAttribute(lyra2_gpu_hash_32_2_shared_4way, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
#endif

	cudaMemset(d_GNonce[thr_id], 0, 2 * sizeof(uint32_t));

	lyra2_gpu_hash_80_1 << < grid, block1 >> > (threads, startNounce, timeCost, nRows, nCols, bug);
	switch (Mode[thr_id])
	{
	case SHARED_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			lyra2_gpu_hash_32_2_shared_4way << < grid, block3, nRows * nCols * sizeof(uint64_t) * 12 * (tpb / 4) >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
		}
		break;
	case GLOBAL_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	case GLOBAL_16WAY_MODE:
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	}
	lyra2_gpu_hash_32_4 << < grid, block1 >> > (threads, startNounce, target, d_GNonce[thr_id], NULL);

	cudaMemcpy(d_gnounce[thr_id], d_GNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	resultnonces[0] = *(d_gnounce[thr_id]);
	resultnonces[1] = *(d_gnounce[thr_id] + 1);
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint64_t *Hash)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	const uint32_t tpb = TPB;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block1(tpb);
	dim3 block2(4, 4, tpb / 16);
	dim3 block3(4, tpb / 4);
	uint32_t sm = 0;
	if (device_sm[dev_id] < 300) sm = 4 * sizeof(uint2)*tpb;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	if (device_sm[dev_id] == 700)
		cudaFuncSetAttribute(lyra2_gpu_hash_32_2_shared_4way, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
#endif

	lyra2_gpu_hash_32_1 << < grid, block1 >> > (threads, startNounce, timeCost, nRows, nCols, bug, (uint2*)Hash);
	switch (Mode[thr_id])
	{
	case SHARED_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			lyra2_gpu_hash_32_2_shared_4way << < grid, block3, nRows * nCols * sizeof(uint64_t) * 12 * (tpb / 4) >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
		}
		break;
	case GLOBAL_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	case GLOBAL_16WAY_MODE:
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	}
	lyra2_gpu_hash_32_4 << < grid, block1 >> > (threads, startNounce, 0, NULL, (uint2*)Hash);
}

__host__
void lyra2_cpu_hash_32_ending(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint64_t *Hash, uint32_t *resultnonces)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	const uint32_t tpb = TPB;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block1(tpb);
	dim3 block2(4, 4, tpb / 16);
	dim3 block3(4, tpb / 4);
	uint32_t sm = 0;
	if (device_sm[dev_id] < 300) sm = 4 * sizeof(uint2)*tpb;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	if (device_sm[dev_id] == 700)
		cudaFuncSetAttribute(lyra2_gpu_hash_32_2_shared_4way, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
#endif

	cudaMemset(d_GNonce[thr_id], 0, 2 * sizeof(uint32_t));

	lyra2_gpu_hash_32_1 << < grid, block1 >> > (threads, startNounce, timeCost, nRows, nCols, bug, (uint2*)Hash);
	switch (Mode[thr_id])
	{
	case SHARED_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			lyra2_gpu_hash_32_2_shared_4way << < grid, block3, nRows * nCols * sizeof(uint64_t) * 12 * (tpb / 4) >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
		}
		break;
	case GLOBAL_4WAY_MODE:
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_4way << < grid, block3, sm >> > (threads, grid.x * 8 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	case GLOBAL_16WAY_MODE:
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols);
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_16way << < grid, block2, sm >> > (threads, grid.x * 2 * i, timeCost, nRows, nCols, tau, prev);
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
		break;
	}
	lyra2_gpu_hash_32_4 << < grid, block1 >> > (threads, startNounce, target, d_GNonce[thr_id], NULL);

	cudaMemcpy(d_gnounce[thr_id], d_GNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	resultnonces[0] = *(d_gnounce[thr_id]);
	resultnonces[1] = *(d_gnounce[thr_id] + 1);
}

__host__
void lyra2_cpu_free(int thr_id)
{
	cudaFree(d_matrix[thr_id]);

	cudaFree(d_GNonce[thr_id]);
	cudaFreeHost(d_gnounce[thr_id]);
}

__host__
void lyra2_setData(const void *data)
{
	cudaMemcpyToSymbol(c_data, data, 80, 0, cudaMemcpyHostToDevice);
}
