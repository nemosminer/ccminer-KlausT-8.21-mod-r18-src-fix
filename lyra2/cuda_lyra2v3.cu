/*
* Lyra2 (v2) CUDA Implementation
*
* Based on tpruvot/djm34/VTC/KlausT sources
*/

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_lyra2_vectors.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */

#ifdef __CUDA_ARCH__
#undef __CUDA_ARCH__
#endif

#define __CUDA_ARCH__ 210

#endif

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
__device__ void __threadfence_block();

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
__device__ void __syncwarp(uint32_t mask);
#endif
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 9000
#define __shfl_sync(mask, a, b, c) __shfl(a, b, c)
#define __syncwarp(mask) __threadfence_block()
#endif

// Max TPB 1024 (Fermi or later)
#define TPB70 64
#define TPB61 128
#define TPB60 64
#define TPB52 128
#define TPB50 128
#define TPB30 192
#define TPB20 96

// Max BPM 32 (Maxwell or later)
// Max BPM 16 (Kepler)
// Max BPM 8 (Fermi or older)
#define BPM70 6
#define BPM61 4
#define BPM60 6
#define BPM52 4
#define BPM50 4
#define BPM30 2
#define BPM20 2

#define BPM2_70 20
#define BPM2_61 10
#define BPM2_60 20
#define BPM2_52 10
#define BPM2_50 10
#define BPM2_30 6
#define BPM2_20 6

#if __CUDA_ARCH__ >= 700
#define TPB TPB70
#define BPM BPM70
#define TPB2 TPB
#define BPM2 BPM2_70
#define REG_MODE
#elif __CUDA_ARCH__ > 600
#define TPB TPB61
#define BPM BPM61
#define TPB2 TPB
#define BPM2 BPM2_61
#define HALF_MODE
#elif __CUDA_ARCH__ == 600
#define TPB TPB60
#define BPM BPM60
#define TPB2 TPB
#define BPM2 BPM2_60
#define REG_MODE
#elif __CUDA_ARCH__ >= 520
#define TPB TPB52
#define BPM BPM52
#define TPB2 TPB
#define BPM2 BPM2_52
#define HALF_MODE
#elif __CUDA_ARCH__ >= 500
#define TPB TPB50
#define BPM BPM50
#define TPB2 TPB
#define BPM2 BPM2_50
#elif __CUDA_ARCH__ >= 300
#define TPB TPB30
#define BPM BPM30
#define TPB2 TPB
#define BPM2 BPM2_30
#define REG_MODE
#else
#define TPB TPB20
#define BPM BPM20
#define TPB2 TPB
#define BPM2 BPM2_20
#define REG_MODE
#endif

__device__ uint2x4 *DState;

#if __CUDA_ARCH__ >= 300
#define WarpShuffle(a, b, c) SHFL(a, b, c)

#define WarpShuffle3(d0, d1, d2, a0, a1, a2, b0, b1, b2, c) { \
	d0.x = SHFL(a0.x, b0, c); \
	d0.y = SHFL(a0.y, b0, c); \
	d1.x = SHFL(a1.x, b1, c); \
	d1.y = SHFL(a1.y, b1, c); \
	d2.x = SHFL(a2.x, b2, c); \
	d2.y = SHFL(a2.y, b2, c); \
}
#else
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))
static __device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	shared_mem[0 * TPB + threadIdx.x] = a;
	__threadfence();
	return shared_mem[0 * TPB + bitselect(threadIdx.x, b, c - 1)];
}

#define WarpShuffle3(d0, d1, d2, a0, a1, a2, b0, b1, b2, c) { \
	extern __shared__ uint32_t shared_mem[]; \
	shared_mem[0 * TPB + threadIdx.x] = a0.x; \
	shared_mem[1 * TPB + threadIdx.x] = a0.y; \
	shared_mem[2 * TPB + threadIdx.x] = a1.x; \
	shared_mem[3 * TPB + threadIdx.x] = a1.y; \
	shared_mem[4 * TPB + threadIdx.x] = a2.x; \
	shared_mem[5 * TPB + threadIdx.x] = a2.y; \
	__threadfence(); \
	d0.x = shared_mem[0 * TPB + bitselect(threadIdx.x, b0, c - 1)]; \
	d0.y = shared_mem[1 * TPB + bitselect(threadIdx.x, b0, c - 1)]; \
	d1.x = shared_mem[2 * TPB + bitselect(threadIdx.x, b1, c - 1)]; \
	d1.y = shared_mem[3 * TPB + bitselect(threadIdx.x, b1, c - 1)]; \
	d2.x = shared_mem[4 * TPB + bitselect(threadIdx.x, b2, c - 1)]; \
	d2.y = shared_mem[5 * TPB + bitselect(threadIdx.x, b2, c - 1)]; \
}
#endif

#define Gfunc(a, b, c, d) { \
	a += b; uint32_t tmp = d.x ^ a.x; d.x = d.y ^ a.y; d.y = tmp; \
	c += d; b ^= c; b = ROR24(b); \
	a += b; d ^= a; d = ROR16(d); \
	c += d; b ^= c; b = ROR2(b, 63); \
}

#define round_lyra_4way(ds0, ds1, ds2, ds3) { \
	Gfunc(ds0, ds1, ds2, ds3); \
	WarpShuffle3(ds1, ds2, ds3, ds1, ds2, ds3, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4); \
	Gfunc(ds0, ds1, ds2, ds3); \
	WarpShuffle3(ds1, ds2, ds3, ds1, ds2, ds3, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4); \
}

#define round_lyra(s0, s1, s2, s3) { \
	Gfunc(s0.x, s1.x, s2.x, s3.x); \
	Gfunc(s0.y, s1.y, s2.y, s3.y); \
	Gfunc(s0.z, s1.z, s2.z, s3.z); \
	Gfunc(s0.w, s1.w, s2.w, s3.w); \
	Gfunc(s0.x, s1.y, s2.z, s3.w); \
	Gfunc(s0.y, s1.z, s2.w, s3.x); \
	Gfunc(s0.z, s1.w, s2.x, s3.y); \
	Gfunc(s0.w, s1.x, s2.y, s3.z); \
}

#define reduceDuplexRowSetup0(s0, n) { \
	s0[n * 3 + 0] = state[0]; \
	s0[n * 3 + 1] = state[1]; \
	s0[n * 3 + 2] = state[2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

#define reduceDuplexRowSetup1(s0, s1, n) { \
	state[0] ^= s0[n * 3 + 0]; \
	state[1] ^= s0[n * 3 + 1]; \
	state[2] ^= s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	s1[n * 3 + 0] = s0[n * 3 + 0] ^ state[0]; \
	s1[n * 3 + 1] = s0[n * 3 + 1] ^ state[1]; \
	s1[n * 3 + 2] = s0[n * 3 + 2] ^ state[2]; \
}

#define reduceDuplexRowSetup2(s0, s1, s2, n, r) { \
	state[0] ^= s1[n * 3 + 0] + s0[n * 3 + 0]; \
	state[1] ^= s1[n * 3 + 1] + s0[n * 3 + 1]; \
	state[2] ^= s1[n * 3 + 2] + s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	s0[n * 3 + 0] ^= index == 0 ? Data2 : Data0; \
	s0[n * 3 + 1] ^= index == 0 ? Data0 : Data1; \
	s0[n * 3 + 2] ^= index == 0 ? Data1 : Data2; \
	s2[n * 3 + 0] = s1[n * 3 + 0] ^ state[0]; \
	s2[n * 3 + 1] = s1[n * 3 + 1] ^ state[1]; \
	s2[n * 3 + 2] = s1[n * 3 + 2] ^ state[2]; \
	shared_mem[(r * 6 + (1 - n) * 3 + 0) * TPB + threadIdx.x] = s2[n * 3 + 0]; \
	shared_mem[(r * 6 + (1 - n) * 3 + 1) * TPB + threadIdx.x] = s2[n * 3 + 1]; \
	shared_mem[(r * 6 + (1 - n) * 3 + 2) * TPB + threadIdx.x] = s2[n * 3 + 2]; \
}

#define reduceDuplexRowSetup3(s0, s1, s2, n, r) { \
	state[0] ^= s1[n * 3 + 0] + s0[n * 3 + 0]; \
	state[1] ^= s1[n * 3 + 1] + s0[n * 3 + 1]; \
	state[2] ^= s1[n * 3 + 2] + s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	s0[n * 3 + 0] ^= index == 0 ? Data2 : Data0; \
	s0[n * 3 + 1] ^= index == 0 ? Data0 : Data1; \
	s0[n * 3 + 2] ^= index == 0 ? Data1 : Data2; \
	shared_mem[(r * 6 + (1 - n) * 3 + 0) * TPB + threadIdx.x] = s0[n * 3 + 0]; \
	shared_mem[(r * 6 + (1 - n) * 3 + 1) * TPB + threadIdx.x] = s0[n * 3 + 1]; \
	shared_mem[(r * 6 + (1 - n) * 3 + 2) * TPB + threadIdx.x] = s0[n * 3 + 2]; \
	s2[n * 3 + 0] = s1[n * 3 + 0] ^ state[0]; \
	s2[n * 3 + 1] = s1[n * 3 + 1] ^ state[1]; \
	s2[n * 3 + 2] = s1[n * 3 + 2] ^ state[2]; \
}

#define reduceDuplexRowt_0(s_in, rowa, s_out, s0, s1, s2, s3, n) { \
	tmp1[3] = rowa == 0 ? s0[n * 3 + 0] : s1[n * 3 + 0]; \
	tmp1[0] = rowa == 2 ? s2[n * 3 + 0] : s3[n * 3 + 0]; \
	if(rowa < 2) tmp1[0] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? s0[n * 3 + 1] : s1[n * 3 + 1]; \
	tmp1[1] = rowa == 2 ? s2[n * 3 + 1] : s3[n * 3 + 1]; \
	if(rowa < 2) tmp1[1] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? s0[n * 3 + 2] : s1[n * 3 + 2]; \
	tmp1[2] = rowa == 2 ? s2[n * 3 + 2] : s3[n * 3 + 2]; \
	if(rowa < 2) tmp1[2] = tmp1[3]; \
	state[0] ^= tmp1[0] + s_in[n * 3 + 0]; \
	state[1] ^= tmp1[1] + s_in[n * 3 + 1]; \
	state[2] ^= tmp1[2] + s_in[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	if(rowa == 0) s0[n * 3 + 0] = tmp1[0]; \
	if(rowa == 0) s0[n * 3 + 1] = tmp1[1]; \
	if(rowa == 0) s0[n * 3 + 2] = tmp1[2]; \
	if(rowa == 1) s1[n * 3 + 0] = tmp1[0]; \
	if(rowa == 1) s1[n * 3 + 1] = tmp1[1]; \
	if(rowa == 1) s1[n * 3 + 2] = tmp1[2]; \
	if(rowa == 2) s2[n * 3 + 0] = tmp1[0]; \
	if(rowa == 2) s2[n * 3 + 1] = tmp1[1]; \
	if(rowa == 2) s2[n * 3 + 2] = tmp1[2]; \
	if(rowa == 3) s3[n * 3 + 0] = tmp1[0]; \
	if(rowa == 3) s3[n * 3 + 1] = tmp1[1]; \
	if(rowa == 3) s3[n * 3 + 2] = tmp1[2]; \
	s_out[n * 3 + 0] ^= state[0]; \
	s_out[n * 3 + 1] ^= state[1]; \
	s_out[n * 3 + 2] ^= state[2]; \
}

#define reduceDuplexRowt_1(rowa, rowout, n) { \
	tmp1[0] = shared_mem[(rowa * 6 + n * 3 + 0) * TPB + threadIdx.x]; \
	tmp1[1] = shared_mem[(rowa * 6 + n * 3 + 1) * TPB + threadIdx.x]; \
	tmp1[2] = shared_mem[(rowa * 6 + n * 3 + 2) * TPB + threadIdx.x]; \
	state[0] ^= tmp0[(1 - n) * 3 + 0] + tmp1[0]; \
	state[1] ^= tmp0[(1 - n) * 3 + 1] + tmp1[1]; \
	state[2] ^= tmp0[(1 - n) * 3 + 2] + tmp1[2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	shared_mem[(rowa * 6 + n * 3 + 0) * TPB + threadIdx.x] = tmp1[0]; \
	shared_mem[(rowa * 6 + n * 3 + 1) * TPB + threadIdx.x] = tmp1[1]; \
	shared_mem[(rowa * 6 + n * 3 + 2) * TPB + threadIdx.x] = tmp1[2]; \
	tmp0[(1 - n) * 3 + 0] = shared_mem[(rowout * 6 + n * 3 + 0) * TPB + threadIdx.x] ^ state[0]; \
	tmp0[(1 - n) * 3 + 1] = shared_mem[(rowout * 6 + n * 3 + 1) * TPB + threadIdx.x] ^ state[1]; \
	tmp0[(1 - n) * 3 + 2] = shared_mem[(rowout * 6 + n * 3 + 2) * TPB + threadIdx.x] ^ state[2]; \
	shared_mem[(rowout * 6 + n * 3 + 0) * TPB + threadIdx.x] = tmp0[(1 - n) * 3 + 0]; \
	shared_mem[(rowout * 6 + n * 3 + 1) * TPB + threadIdx.x] = tmp0[(1 - n) * 3 + 1]; \
	shared_mem[(rowout * 6 + n * 3 + 2) * TPB + threadIdx.x] = tmp0[(1 - n) * 3 + 2]; \
}

#define reduceDuplexRowt_2(s_in, rowa, s0, s1, s2, s3, n) { \
	tmp1[3] = rowa == 0 ? s0[n * 3 + 0] : s1[n * 3 + 0]; \
	tmp1[0] = rowa == 2 ? s2[n * 3 + 0] : s3[n * 3 + 0]; \
	if(rowa < 2) tmp1[0] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? s0[n * 3 + 1] : s1[n * 3 + 1]; \
	tmp1[1] = rowa == 2 ? s2[n * 3 + 1] : s3[n * 3 + 1]; \
	if(rowa < 2) tmp1[1] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? s0[n * 3 + 2] : s1[n * 3 + 2]; \
	tmp1[2] = rowa == 2 ? s2[n * 3 + 2] : s3[n * 3 + 2]; \
	if(rowa < 2) tmp1[2] = tmp1[3]; \
	state[0] ^= tmp1[0] + s_in[n * 3 + 0]; \
	state[1] ^= tmp1[1] + s_in[n * 3 + 1]; \
	state[2] ^= tmp1[2] + s_in[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	if(rowa == 3) tmp1[0] ^= state[0]; \
	if(rowa == 3) tmp1[1] ^= state[1]; \
	if(rowa == 3) tmp1[2] ^= state[2]; \
}

#define reduceDuplexRowt_3(s_in, rowa, s0, s1, s2, s3, n) { \
	tmp1[4] = rowa == 0 ? s0[n * 3 + 0] : s1[n * 3 + 0]; \
	tmp1[3] = rowa == 2 ? s2[n * 3 + 0] : s3[n * 3 + 0]; \
	if(rowa < 2) tmp1[3] = tmp1[4]; \
	state[0] ^= tmp1[3] + s_in[n * 3 + 0]; \
	tmp1[4] = rowa == 0 ? s0[n * 3 + 1] : s1[n * 3 + 1]; \
	tmp1[3] = rowa == 2 ? s2[n * 3 + 1] : s3[n * 3 + 1]; \
	if(rowa < 2) tmp1[3] = tmp1[4]; \
	state[1] ^= tmp1[3] + s_in[n * 3 + 1]; \
	tmp1[4] = rowa == 0 ? s0[n * 3 + 2] : s1[n * 3 + 2]; \
	tmp1[3] = rowa == 2 ? s2[n * 3 + 2] : s3[n * 3 + 2]; \
	if(rowa < 2) tmp1[3] = tmp1[4]; \
	state[2] ^= tmp1[3] + s_in[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

#define reduceDuplexRowt_4(rowa, n) { \
	tmp1[3] = shared_mem[(rowa * 6 + n * 3 + 0) * TPB + threadIdx.x]; \
	tmp1[4] = shared_mem[(rowa * 6 + n * 3 + 1) * TPB + threadIdx.x]; \
	tmp1[5] = shared_mem[(rowa * 6 + n * 3 + 2) * TPB + threadIdx.x]; \
	state[0] ^= tmp0[(1 - n) * 3 + 0] + tmp1[3]; \
	state[1] ^= tmp0[(1 - n) * 3 + 1] + tmp1[4]; \
	state[2] ^= tmp0[(1 - n) * 3 + 2] + tmp1[5]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

#define reduceDuplexRowSetup0_v60(ds0, ds1, dn0, dn1) { \
	ds0[dn0 * 2 + 0] = state[0]; \
	ds0[dn0 * 2 + 1] = state[1]; \
	ds1[dn1] = state[2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

#define reduceDuplexRowSetup1_v60(ss0, ss1, sn0, sn1, ds0, ds1, dn0, dn1) { \
	state[0] ^= ss0[sn0 * 2 + 0]; \
	state[1] ^= ss0[sn0 * 2 + 1]; \
	state[2] ^= ss1[sn1]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	ds0[dn0 * 2 + 0] = ss0[sn0 * 2 + 0] ^ state[0]; \
	ds0[dn0 * 2 + 1] = ss0[sn0 * 2 + 1] ^ state[1]; \
	ds1[dn1] = ss1[sn1] ^ state[2]; \
}

#define reduceDuplexRowSetup2_v60(ss0, ss1, sn0, sn1, ss2, ss3, sn2, sn3, ds0, ds1, dn0, dn1, r0, r1) { \
	state[0] ^= ss2[sn2 * 2 + 0] + ss0[sn0 * 2 + 0]; \
	state[1] ^= ss2[sn2 * 2 + 1] + ss0[sn0 * 2 + 1]; \
	state[2] ^= ss3[sn3] + ss1[sn1]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	ss0[sn0 * 2 + 0] ^= index == 0 ? Data2 : Data0; \
	ss0[sn0 * 2 + 1] ^= index == 0 ? Data0 : Data1; \
	ss1[sn1] ^= index == 0 ? Data1 : Data2; \
	shared_mem[(r0 * 4 + sn0) * TPB + threadIdx.x] = ss1[sn1]; \
	ds1[dn1] = ss3[sn3] ^ state[2]; \
	ds0[dn0 * 2 + 0] = ss2[sn2 * 2 + 0] ^ state[0]; \
	ds0[dn0 * 2 + 1] = ss2[sn2 * 2 + 1] ^ state[1]; \
	shared_mem[(r1 * 4 + dn0) * TPB + threadIdx.x] = ds1[dn1]; \
}

#define reduceDuplexRowt_0_v60(s0, s1, s2, n0, r1, r2) { \
	tmp1[2] = rowa == 0 ? state0[n0 * 2 + 0] : state1[n0 * 2 + 0]; \
	tmp1[0] = rowa == 2 ? state2[n0 * 2 + 0] : state3[n0 * 2 + 0]; \
	if(rowa < 2) tmp1[0] = tmp1[2]; \
	tmp1[2] = rowa == 0 ? state0[n0 * 2 + 1] : state1[n0 * 2 + 1]; \
	tmp1[1] = rowa == 2 ? state2[n0 * 2 + 1] : state3[n0 * 2 + 1]; \
	if(rowa < 2) tmp1[1] = tmp1[2]; \
	tmp1[2] = shared_mem[(r1 * 4 + n0) * TPB + threadIdx.x]; \
	state[0] ^= tmp1[0] + s0[n0 * 2 + 0]; \
	state[1] ^= tmp1[1] + s0[n0 * 2 + 1]; \
	state[2] ^= tmp1[2] + s1[n0]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	if(rowa == 0) state0[n0 * 2 + 0] = tmp1[0]; \
	if(rowa == 0) state0[n0 * 2 + 1] = tmp1[1]; \
	if(rowa == 1) state1[n0 * 2 + 0] = tmp1[0]; \
	if(rowa == 1) state1[n0 * 2 + 1] = tmp1[1]; \
	if(rowa == 2) state2[n0 * 2 + 0] = tmp1[0]; \
	if(rowa == 2) state2[n0 * 2 + 1] = tmp1[1]; \
	if(rowa == 3) state3[n0 * 2 + 0] = tmp1[0]; \
	if(rowa == 3) state3[n0 * 2 + 1] = tmp1[1]; \
	shared_mem[(r1 * 4 + n0) * TPB + threadIdx.x] = tmp1[2]; \
	s2[n0 * 2 + 0] ^= state[0]; \
	s2[n0 * 2 + 1] ^= state[1]; \
	s1[n0] = shared_mem[(r2 * 4 + n0) * TPB + threadIdx.x] ^ state[2]; \
	shared_mem[(r2 * 4 + n0) * TPB + threadIdx.x] = s1[n0]; \
}

#define reduceDuplexRowt_1_v60(s0, s1, n0, r1) { \
	tmp1[2] = rowa == 0 ? state0[n0 * 2 + 0] : state1[n0 * 2 + 0]; \
	tmp1[0] = rowa == 2 ? state2[n0 * 2 + 0] : state3[n0 * 2 + 0]; \
	if(rowa < 2) tmp1[0] = tmp1[2]; \
	tmp1[2] = rowa == 0 ? state0[n0 * 2 + 1] : state1[n0 * 2 + 1]; \
	tmp1[1] = rowa == 2 ? state2[n0 * 2 + 1] : state3[n0 * 2 + 1]; \
	if(rowa < 2) tmp1[1] = tmp1[2]; \
	tmp1[2] = shared_mem[(r1 * 4 + n0) * TPB + threadIdx.x]; \
	state[0] ^= tmp1[0] + s0[n0 * 2 + 0]; \
	state[1] ^= tmp1[1] + s0[n0 * 2 + 1]; \
	state[2] ^= tmp1[2] + s1[n0]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	if(rowa == 3) tmp1[0] ^= state[0]; \
	if(rowa == 3) tmp1[1] ^= state[1]; \
	if(rowa == 3) tmp1[2] ^= state[2]; \
}

#define reduceDuplexRowt_2_v60(s0, s1, n0, r1) { \
	uint2 Data0, Data1, Data2; \
	Data2 = rowa == 0 ? state0[n0 * 2 + 0] : state1[n0 * 2 + 0]; \
	Data0 = rowa == 2 ? state2[n0 * 2 + 0] : state3[n0 * 2 + 0]; \
	if(rowa < 2) Data0 = Data2; \
	Data2 = rowa == 0 ? state0[n0 * 2 + 1] : state1[n0 * 2 + 1]; \
	Data1 = rowa == 2 ? state2[n0 * 2 + 1] : state3[n0 * 2 + 1]; \
	if(rowa < 2) Data1 = Data2; \
	Data2 = shared_mem[(r1 * 4 + n0) * TPB + threadIdx.x]; \
	state[0] ^= Data0 + s0[n0 * 2 + 0]; \
	state[1] ^= Data1 + s0[n0 * 2 + 1]; \
	state[2] ^= Data2 + s1[n0]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

#define reduceDuplexRowSetup0_v30(s0, n) { \
	s0[(3 - n) * 3 + 0] = state[0]; \
	s0[(3 - n) * 3 + 1] = state[1]; \
	s0[(3 - n) * 3 + 2] = state[2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

#define reduceDuplexRowSetup1_v30(s0, s1, n) { \
	state[0] ^= s0[n * 3 + 0]; \
	state[1] ^= s0[n * 3 + 1]; \
	state[2] ^= s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	s1[(3 - n) * 3 + 0] = s0[n * 3 + 0] ^ state[0]; \
	s1[(3 - n) * 3 + 1] = s0[n * 3 + 1] ^ state[1]; \
	s1[(3 - n) * 3 + 2] = s0[n * 3 + 2] ^ state[2]; \
}

#define reduceDuplexRowSetup2_v30(s0, s1, s2, n) { \
	state[0] ^= s1[n * 3 + 0] + s0[n * 3 + 0]; \
	state[1] ^= s1[n * 3 + 1] + s0[n * 3 + 1]; \
	state[2] ^= s1[n * 3 + 2] + s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	s0[n * 3 + 0] ^= index == 0 ? Data2 : Data0; \
	s0[n * 3 + 1] ^= index == 0 ? Data0 : Data1; \
	s0[n * 3 + 2] ^= index == 0 ? Data1 : Data2; \
	s2[(3 - n) * 3 + 0] = s1[n * 3 + 0] ^ state[0]; \
	s2[(3 - n) * 3 + 1] = s1[n * 3 + 1] ^ state[1]; \
	s2[(3 - n) * 3 + 2] = s1[n * 3 + 2] ^ state[2]; \
}

#define reduceDuplexRowt_0_v30(s0, s1, n) { \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 0] : state1[n * 3 + 0]; \
	tmp1[0] = rowa == 2 ? state2[n * 3 + 0] : state3[n * 3 + 0]; \
	if(rowa < 2) tmp1[0] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 1] : state1[n * 3 + 1]; \
	tmp1[1] = rowa == 2 ? state2[n * 3 + 1] : state3[n * 3 + 1]; \
	if(rowa < 2) tmp1[1] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 2] : state1[n * 3 + 2]; \
	tmp1[2] = rowa == 2 ? state2[n * 3 + 2] : state3[n * 3 + 2]; \
	if(rowa < 2) tmp1[2] = tmp1[3]; \
	state[0] ^= tmp1[0] + s0[n * 3 + 0]; \
	state[1] ^= tmp1[1] + s0[n * 3 + 1]; \
	state[2] ^= tmp1[2] + s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	if(rowa == 0) state0[n * 3 + 0] = tmp1[0]; \
	if(rowa == 0) state0[n * 3 + 1] = tmp1[1]; \
	if(rowa == 0) state0[n * 3 + 2] = tmp1[2]; \
	if(rowa == 1) state1[n * 3 + 0] = tmp1[0]; \
	if(rowa == 1) state1[n * 3 + 1] = tmp1[1]; \
	if(rowa == 1) state1[n * 3 + 2] = tmp1[2]; \
	if(rowa == 2) state2[n * 3 + 0] = tmp1[0]; \
	if(rowa == 2) state2[n * 3 + 1] = tmp1[1]; \
	if(rowa == 2) state2[n * 3 + 2] = tmp1[2]; \
	if(rowa == 3) state3[n * 3 + 0] = tmp1[0]; \
	if(rowa == 3) state3[n * 3 + 1] = tmp1[1]; \
	if(rowa == 3) state3[n * 3 + 2] = tmp1[2]; \
	s1[n * 3 + 0] ^= state[0]; \
	s1[n * 3 + 1] ^= state[1]; \
	s1[n * 3 + 2] ^= state[2]; \
}

#define reduceDuplexRowt_1_v30(s0, n) { \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 0] : state1[n * 3 + 0]; \
	tmp1[0] = rowa == 2 ? state2[n * 3 + 0] : state3[n * 3 + 0]; \
	if(rowa < 2) tmp1[0] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 1] : state1[n * 3 + 1]; \
	tmp1[1] = rowa == 2 ? state2[n * 3 + 1] : state3[n * 3 + 1]; \
	if(rowa < 2) tmp1[1] = tmp1[3]; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 2] : state1[n * 3 + 2]; \
	tmp1[2] = rowa == 2 ? state2[n * 3 + 2] : state3[n * 3 + 2]; \
	if(rowa < 2) tmp1[2] = tmp1[3]; \
	state[0] ^= tmp1[0] + s0[n * 3 + 0]; \
	state[1] ^= tmp1[1] + s0[n * 3 + 1]; \
	state[2] ^= tmp1[2] + s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
	uint2 Data0, Data1, Data2; \
	WarpShuffle3(Data0, Data1, Data2, state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4); \
	tmp1[0] ^= index == 0 ? Data2 : Data0; \
	tmp1[1] ^= index == 0 ? Data0 : Data1; \
	tmp1[2] ^= index == 0 ? Data1 : Data2; \
	if(rowa == 3) tmp1[0] ^= state[0]; \
	if(rowa == 3) tmp1[1] ^= state[1]; \
	if(rowa == 3) tmp1[2] ^= state[2]; \
}

#define reduceDuplexRowt_2_v30(s0, n) { \
	uint2 Data0, Data1, Data2; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 0] : state1[n * 3 + 0]; \
	Data0 = rowa == 2 ? state2[n * 3 + 0] : state3[n * 3 + 0]; \
	if(rowa < 2) Data0 = tmp1[3]; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 1] : state1[n * 3 + 1]; \
	Data1 = rowa == 2 ? state2[n * 3 + 1] : state3[n * 3 + 1]; \
	if(rowa < 2) Data1 = tmp1[3]; \
	tmp1[3] = rowa == 0 ? state0[n * 3 + 2] : state1[n * 3 + 2]; \
	Data2 = rowa == 2 ? state2[n * 3 + 2] : state3[n * 3 + 2]; \
	if(rowa < 2) Data2 = tmp1[3]; \
	state[0] ^= Data0 + s0[n * 3 + 0]; \
	state[1] ^= Data1 + s0[n * 3 + 1]; \
	state[2] ^= Data2 + s0[n * 3 + 2]; \
	round_lyra_4way(state[0], state[1], state[2], state[3]); \
}

__global__ __launch_bounds__(TPB2, BPM2)
void lyra2v3_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockIdx.x * TPB2 + threadIdx.x;

	if (thread < threads) {
		uint28 state[4];
		
		state[0].x = state[1].x = __ldg(&outputHash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&outputHash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&outputHash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&outputHash[thread + threads * 3]);
		state[2].x.x = 0xf3bcc908lu;
		state[2].x.y = 0x6a09e667lu;
		state[2].y.x = 0x84caa73blu;
		state[2].y.y = 0xbb67ae85lu;
		state[2].z.x = 0xfe94f82blu;
		state[2].z.y = 0x3c6ef372lu;
		state[2].w.x = 0x5f1d36f1lu;
		state[2].w.y = 0xa54ff53alu;
		state[3].x.x = 0xade682d1lu;
		state[3].x.y = 0x510e527flu;
		state[3].y.x = 0x2b3e6c1flu;
		state[3].y.y = 0x9b05688clu;
		state[3].z.x = 0xfb41bd6blu;
		state[3].z.y = 0x1f83d9ablu;
		state[3].w.x = 0x137e2179lu;
		state[3].w.y = 0x5be0cd19lu;

		for (uint32_t i = 0; i < 12; i++)
			round_lyra(state[0], state[1], state[2], state[3]);

		state[0].x.x ^= 0x00000020lu;
		state[0].y.x ^= 0x00000020lu;
		state[0].z.x ^= 0x00000020lu;
		state[0].w.x ^= 0x00000001lu;
		state[1].x.x ^= 0x00000004lu;
		state[1].y.x ^= 0x00000004lu;
		state[1].z.x ^= 0x00000080lu;
		state[1].w.y ^= 0x01000000lu;

		for (uint32_t i = 0; i < 12; i++)
			round_lyra(state[0], state[1], state[2], state[3]);

		DState[threads * 0 + thread] = state[0];
		DState[threads * 1 + thread] = state[1];
		DState[threads * 2 + thread] = state[2];
		DState[threads * 3 + thread] = state[3];
	}
}

__global__ __launch_bounds__(TPB, BPM)
void lyra2v3_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockIdx.x * TPB + threadIdx.x;

	if (thread < threads * 4) {
		const uint32_t index = threadIdx.x & 3;
		uint2 state[4];
		uint32_t rowa;

		state[0] = __ldg(&((uint2*)DState)[threads * 4 * 0 + thread]);
		state[1] = __ldg(&((uint2*)DState)[threads * 4 * 1 + thread]);
		state[2] = __ldg(&((uint2*)DState)[threads * 4 * 2 + thread]);
		state[3] = __ldg(&((uint2*)DState)[threads * 4 * 3 + thread]);

#if defined(HALF_MODE)
		extern __shared__ uint2 shared_mem[];

		uint2 state0[6], state1[6], state2[6], state3[6];
		uint2 tmp0[6], tmp1[6];

		// RowSetup_0
		reduceDuplexRowSetup0(state2, 0);
		reduceDuplexRowSetup0(state2, 1);
		reduceDuplexRowSetup0(state0, 1);
		reduceDuplexRowSetup0(state0, 0);

		// RowSetup_1
		reduceDuplexRowSetup1(state0, state3, 0);
		reduceDuplexRowSetup1(state0, state3, 1);
		reduceDuplexRowSetup1(state2, state1, 1);
		reduceDuplexRowSetup1(state2, state1, 0);

		// RowSetup_2
		reduceDuplexRowSetup2(state0, state1, tmp1, 0, 2);
		reduceDuplexRowSetup2(state0, state1, tmp1, 1, 2);
		reduceDuplexRowSetup3(state2, state3, state2, 1, 0);
		reduceDuplexRowSetup3(state2, state3, state2, 0, 0);

		// RowSetup_3
		reduceDuplexRowSetup2(state1, state2, tmp0, 0, 3);
		reduceDuplexRowSetup2(state1, state2, tmp0, 1, 3);
		reduceDuplexRowSetup3(state3, tmp1, state3, 1, 1);
		reduceDuplexRowSetup3(state3, tmp1, state3, 0, 1);

		// reduceDuplex_0
		uint32_t instance = WarpShuffle(state[0].x & 15, 0, 4);
		uint32_t buf0, buf1;
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0(state3, rowa, state0, state0, state1, state2, state3, 0);
		reduceDuplexRowt_0(state3, rowa, state0, state0, state1, state2, state3, 1);
		reduceDuplexRowt_1(rowa, 0, 0);
		reduceDuplexRowt_1(rowa, 0, 1);

		// reduceDuplex_1
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0(state0, rowa, state1, state0, state1, state2, state3, 0);
		reduceDuplexRowt_0(state0, rowa, state1, state0, state1, state2, state3, 1);
		reduceDuplexRowt_1(rowa, 1, 0);
		reduceDuplexRowt_1(rowa, 1, 1);

		// reduceDuplex_2
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0(state1, rowa, state2, state0, state1, state2, state3, 0);
		reduceDuplexRowt_0(state1, rowa, state2, state0, state1, state2, state3, 1);
		reduceDuplexRowt_1(rowa, 2, 0);
		reduceDuplexRowt_1(rowa, 2, 1);

		// reduceDuplex_3
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_2(state2, rowa, state0, state1, state2, state3, 0);
		reduceDuplexRowt_3(state2, rowa, state0, state1, state2, state3, 1);
		reduceDuplexRowt_4(rowa, 0);
		reduceDuplexRowt_4(rowa, 1);
#elif defined(REG_MODE)

		uint2 state0[12], state1[12], state2[12], state3[12];
		uint2 tmp1[4];

		// RowSetup_0
		reduceDuplexRowSetup0_v30(state0, 0);
		reduceDuplexRowSetup0_v30(state0, 1);
		reduceDuplexRowSetup0_v30(state0, 2);
		reduceDuplexRowSetup0_v30(state0, 3);

		// RowSetup_1
		reduceDuplexRowSetup1_v30(state0, state1, 0);
		reduceDuplexRowSetup1_v30(state0, state1, 1);
		reduceDuplexRowSetup1_v30(state0, state1, 2);
		reduceDuplexRowSetup1_v30(state0, state1, 3);

		// RowSetup_2
		reduceDuplexRowSetup2_v30(state0, state1, state2, 0);
		reduceDuplexRowSetup2_v30(state0, state1, state2, 1);
		reduceDuplexRowSetup2_v30(state0, state1, state2, 2);
		reduceDuplexRowSetup2_v30(state0, state1, state2, 3);

		// RowSetup_3
		reduceDuplexRowSetup2_v30(state1, state2, state3, 0);
		reduceDuplexRowSetup2_v30(state1, state2, state3, 1);
		reduceDuplexRowSetup2_v30(state1, state2, state3, 2);
		reduceDuplexRowSetup2_v30(state1, state2, state3, 3);

		// reduceDuplex_0
		uint32_t instance = WarpShuffle(state[0].x & 15, 0, 4);
		uint32_t buf0, buf1;
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0_v30(state3, state0, 0);
		reduceDuplexRowt_0_v30(state3, state0, 1);
		reduceDuplexRowt_0_v30(state3, state0, 2);
		reduceDuplexRowt_0_v30(state3, state0, 3);

		// reduceDuplex_1
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0_v30(state0, state1, 0);
		reduceDuplexRowt_0_v30(state0, state1, 1);
		reduceDuplexRowt_0_v30(state0, state1, 2);
		reduceDuplexRowt_0_v30(state0, state1, 3);

		// reduceDuplex_2
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0_v30(state1, state2, 0);
		reduceDuplexRowt_0_v30(state1, state2, 1);
		reduceDuplexRowt_0_v30(state1, state2, 2);
		reduceDuplexRowt_0_v30(state1, state2, 3);

		// reduceDuplex_3
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_1_v30(state2, 0);
		reduceDuplexRowt_2_v30(state2, 1);
		reduceDuplexRowt_2_v30(state2, 2);
		reduceDuplexRowt_2_v30(state2, 3);
#else
		extern __shared__ uint2 shared_mem[];

		uint2 state0[8], state1[8], state2[8], state3[8];
		uint2 tmp0[4], tmp1[3];

		// RowSetup_0
		reduceDuplexRowSetup0_v60(state0, tmp0, 3, 3);
		reduceDuplexRowSetup0_v60(state0, tmp0, 2, 2);
		reduceDuplexRowSetup0_v60(state0, tmp0, 1, 1);
		reduceDuplexRowSetup0_v60(state0, tmp0, 0, 0);

		// RowSetup_1
		reduceDuplexRowSetup1_v60(state0, tmp0, 0, 0, state1, state3, 3, 0);
		reduceDuplexRowSetup1_v60(state0, tmp0, 1, 1, state1, state3, 2, 1);
		reduceDuplexRowSetup1_v60(state0, tmp0, 2, 2, state1, state3, 1, 2);
		reduceDuplexRowSetup1_v60(state0, tmp0, 3, 3, state1, state3, 0, 3);

		// RowSetup_2
		reduceDuplexRowSetup2_v60(state0, tmp0, 0, 0, state1, state3, 0, 3, state2, tmp0, 3, 0, 0, 2);
		reduceDuplexRowSetup2_v60(state0, tmp0, 1, 1, state1, state3, 1, 2, state2, tmp0, 2, 1, 0, 2);
		reduceDuplexRowSetup2_v60(state0, tmp0, 2, 2, state1, state3, 2, 1, state2, tmp0, 1, 2, 0, 2);
		reduceDuplexRowSetup2_v60(state0, tmp0, 3, 3, state1, state3, 3, 0, state2, tmp0, 0, 3, 0, 2);

		// RowSetup_3
		reduceDuplexRowSetup2_v60(state1, state3, 0, 3, state2, tmp0, 0, 3, state3, tmp0, 3, 3, 1, 3);
		reduceDuplexRowSetup2_v60(state1, state3, 1, 2, state2, tmp0, 1, 2, state3, tmp0, 2, 2, 1, 3);
		reduceDuplexRowSetup2_v60(state1, state3, 2, 1, state2, tmp0, 2, 1, state3, tmp0, 1, 1, 1, 3);
		reduceDuplexRowSetup2_v60(state1, state3, 3, 0, state2, tmp0, 3, 0, state3, tmp0, 0, 0, 1, 3);

		// reduceDuplex_0
		uint32_t instance = WarpShuffle(state[0].x & 15, 0, 4);
		uint32_t buf0, buf1;
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0_v60(state3, tmp0, state0, 0, rowa, 0);
		reduceDuplexRowt_0_v60(state3, tmp0, state0, 1, rowa, 0);
		reduceDuplexRowt_0_v60(state3, tmp0, state0, 2, rowa, 0);
		reduceDuplexRowt_0_v60(state3, tmp0, state0, 3, rowa, 0);

		// reduceDuplex_1
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0_v60(state0, tmp0, state1, 0, rowa, 1);
		reduceDuplexRowt_0_v60(state0, tmp0, state1, 1, rowa, 1);
		reduceDuplexRowt_0_v60(state0, tmp0, state1, 2, rowa, 1);
		reduceDuplexRowt_0_v60(state0, tmp0, state1, 3, rowa, 1);

		// reduceDuplex_2
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_0_v60(state1, tmp0, state2, 0, rowa, 2);
		reduceDuplexRowt_0_v60(state1, tmp0, state2, 1, rowa, 2);
		reduceDuplexRowt_0_v60(state1, tmp0, state2, 2, rowa, 2);
		reduceDuplexRowt_0_v60(state1, tmp0, state2, 3, rowa, 2);

		// reduceDuplex_3
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		instance = WarpShuffle(buf0 & 15, instance & 3, 4);
		buf0 = (instance >> 2) < 2 ? state[0].x : state[2].x;
		buf1 = (instance >> 2) < 2 ? state[1].x : state[3].x;
		buf0 = ((instance >> 2) & 1) == 0 ? buf0 : buf1;
		rowa = WarpShuffle(buf0 & 3, instance & 3, 4);
		reduceDuplexRowt_1_v60(state2, tmp0, 0, rowa);
		reduceDuplexRowt_2_v60(state2, tmp0, 1, rowa);
		reduceDuplexRowt_2_v60(state2, tmp0, 2, rowa);
		reduceDuplexRowt_2_v60(state2, tmp0, 3, rowa);
#endif

		// finish
		state[0] ^= tmp1[0];
		state[1] ^= tmp1[1];
		state[2] ^= tmp1[2];

		((uint2*)DState)[threads * 4 * 0 + thread] = state[0];
		((uint2*)DState)[threads * 4 * 1 + thread] = state[1];
		((uint2*)DState)[threads * 4 * 2 + thread] = state[2];
		((uint2*)DState)[threads * 4 * 3 + thread] = state[3];
	}
}

__global__ __launch_bounds__(TPB2, BPM2)
void lyra2v3_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockIdx.x * TPB2 + threadIdx.x;

	if (thread < threads) {
		uint28 state[4];

		state[0] = __ldg4(&DState[threads * 0 + thread]);
		state[1] = __ldg4(&DState[threads * 1 + thread]);
		state[2] = __ldg4(&DState[threads * 2 + thread]);
		state[3] = __ldg4(&DState[threads * 3 + thread]);

		for (uint32_t i = 0; i < 12; i++)
			round_lyra(state[0], state[1], state[2], state[3]);

		outputHash[thread + threads * 0] = state[0].x;
		outputHash[thread + threads * 1] = state[0].y;
		outputHash[thread + threads * 2] = state[0].z;
		outputHash[thread + threads * 3] = state[0].w;
	}
}

__host__
void lyra2v3_cpu_init(int thr_id, uint64_t *d_matrix)
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DState, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2v3_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb1, tpb2, sm;

	if (device_sm[dev_id] >= 750) { tpb1 = TPB70; sm = 0; }
	else if (device_sm[dev_id] >= 610) { tpb1 = TPB61; sm = TPB61 * 192; }
	else if (device_sm[dev_id] >= 600) { tpb1 = TPB60; sm = 0; }
	else if (device_sm[dev_id] >= 520) { tpb1 = TPB52; sm = TPB52 * 192; }
	else if (device_sm[dev_id] >= 500) { tpb1 = TPB50; sm = TPB50 * 128; }
	else if (device_sm[dev_id] >= 300) { tpb1 = TPB30; sm = 0; }
	else { tpb1 = TPB20; sm = TPB50 * 6 * 4; }

	tpb2 = tpb1;

	dim3 grid1((threads * 4 + tpb1 - 1) / tpb1);
	dim3 block1(tpb1);

	dim3 grid2((threads + tpb2 - 1) / tpb2);
	dim3 block2(tpb2);

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	if (device_sm[dev_id] == 700)
		cudaFuncSetAttribute(lyra2v3_gpu_hash_32_2, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
#endif

	lyra2v3_gpu_hash_32_1 << <grid2, block2, 0, gpustream[thr_id] >> > (threads, startNounce, (uint2*)g_hash);
	CUDA_SAFE_CALL(cudaGetLastError());

	lyra2v3_gpu_hash_32_2 << <grid1, block1, sm, gpustream[thr_id] >> > (threads, startNounce, (uint2*)g_hash);
	CUDA_SAFE_CALL(cudaGetLastError());

	lyra2v3_gpu_hash_32_3 << <grid2, block2, 0, gpustream[thr_id] >> > (threads, startNounce, (uint2*)g_hash);
	CUDA_SAFE_CALL(cudaGetLastError());

	//MyStreamSynchronize(NULL, order, thr_id);
}
