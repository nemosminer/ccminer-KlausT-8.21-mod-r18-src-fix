#include <stdio.h>
#include <memory.h>
#include "miner.h"
#include "cuda_helper.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 210
#define asm()
__device__ void __threadfence_block();

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
__device__ void __syncwarp(uint32_t mask);
#endif
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 9000
#define __syncwarp(mask) __threadfence_block()
#endif

#if __CUDA_ARCH__ < 300
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint32_t buf, result;

	__syncwarp(0xFFFFFFFF);
	buf = shared_mem[threads * 0 + thread];
	shared_mem[threads * 0 + thread] = a;
	__syncwarp(0xFFFFFFFF);
	result = shared_mem[0 * threads + bitselect(thread, b, c - 1)];
	__syncwarp(0xFFFFFFFF);
	shared_mem[threads * 0 + thread] = buf;

	return result;
}

__device__ __forceinline__ void WarpShuffle2(uint32_t &d0, uint32_t &d1, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint32_t buf0, buf1;

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

__device__ __forceinline__ void WarpShuffle3(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint32_t buf0, buf1, buf2;

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

__device__ __forceinline__ void WarpShuffle4(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	extern __shared__ uint32_t shared_mem[];
	uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t threads = blockDim.y * blockDim.x;
	uint32_t buf0, buf1, buf2, buf3;

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
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return SHFL(a, b, c);
}

__device__ __forceinline__ void WarpShuffle2(uint32_t &d0, uint32_t &d1, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, uint32_t c)
{
	d0 = SHFL(a0, b0, c);
	d1 = SHFL(a1, b1, c);
}

__device__ __forceinline__ void WarpShuffle3(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	d0 = SHFL(a0, b0, c);
	d1 = SHFL(a1, b1, c);
	d2 = SHFL(a2, b2, c);
}

__device__ __forceinline__ void WarpShuffle4(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	d0 = SHFL(a0, b0, c);
	d1 = SHFL(a1, b1, c);
	d2 = SHFL(a2, b2, c);
	d3 = SHFL(a3, b3, c);
}

#endif

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__device__ static uint32_t *B;
__device__ static uint32_t *S;
__device__ static uint32_t *V;
__device__ static uint32_t *sha256;

static uint32_t *d_gnounce[MAX_GPUS];
static uint32_t *d_GNonce[MAX_GPUS];

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// sha256 function ///////////////////////////////////

__constant__ static uint32_t c_data[28];
__constant__ static uint32_t cpu_h[8];
__constant__ static uint32_t c_K[64];
__constant__ static uint32_t client_key[32];
__constant__ static uint32_t client_key_len[1];

/* Elementary functions used by SHA256 */
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define ROTR(x, n)      ((x >> n) | (x << (32 - n)))
#define ROTL(x, n)      ((x << n) | (x >> (32 - n)))
#define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

/* SHA256 round function */
#define RND(a, b, c, d, e, f, g, h, W, k, i) \
	do { \
		h += S1(e) + Ch(e, f, g) + W[i] + k; \
		d += h; \
		h += S0(a) + Maj(a, b, c); \
	} while (0)

/* Adjusted round function for rotating state */
#define RNDr(a, b, c, d, e, f, g, h, W, k, i) { \
	W[i] += s1(W[i + 14]) + W[i + 9] + s0(W[i + 1]); \
	RND(a, b, c, d, e, f, g, h, W, k, i); \
}

//
// Host code
//
static const uint32_t cpu_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

__host__
static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t in, const uint32_t Kshared)
{
	uint32_t t1, t2;
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_step2_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1, t2;

	int pcidx1 = (pc - 2) & 0xF;
	int pcidx2 = (pc - 7) & 0xF;
	int pcidx3 = (pc - 15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	uint32_t ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_round_body_host(uint32_t* in, uint32_t* state, const uint32_t* Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha256_step1_host(a, b, c, d, e, f, g, h, in[0], cpu_K[0]);
	sha256_step1_host(h, a, b, c, d, e, f, g, in[1], cpu_K[1]);
	sha256_step1_host(g, h, a, b, c, d, e, f, in[2], cpu_K[2]);
	sha256_step1_host(f, g, h, a, b, c, d, e, in[3], cpu_K[3]);
	sha256_step1_host(e, f, g, h, a, b, c, d, in[4], cpu_K[4]);
	sha256_step1_host(d, e, f, g, h, a, b, c, in[5], cpu_K[5]);
	sha256_step1_host(c, d, e, f, g, h, a, b, in[6], cpu_K[6]);
	sha256_step1_host(b, c, d, e, f, g, h, a, in[7], cpu_K[7]);
	sha256_step1_host(a, b, c, d, e, f, g, h, in[8], cpu_K[8]);
	sha256_step1_host(h, a, b, c, d, e, f, g, in[9], cpu_K[9]);
	sha256_step1_host(g, h, a, b, c, d, e, f, in[10], cpu_K[10]);
	sha256_step1_host(f, g, h, a, b, c, d, e, in[11], cpu_K[11]);
	sha256_step1_host(e, f, g, h, a, b, c, d, in[12], cpu_K[12]);
	sha256_step1_host(d, e, f, g, h, a, b, c, in[13], cpu_K[13]);
	sha256_step1_host(c, d, e, f, g, h, a, b, in[14], cpu_K[14]);
	sha256_step1_host(b, c, d, e, f, g, h, a, in[15], cpu_K[15]);

	for (int i = 0; i < 3; i++)
	{
		sha256_step2_host(a, b, c, d, e, f, g, h, in, 0, cpu_K[16 + 16 * i]);
		sha256_step2_host(h, a, b, c, d, e, f, g, in, 1, cpu_K[17 + 16 * i]);
		sha256_step2_host(g, h, a, b, c, d, e, f, in, 2, cpu_K[18 + 16 * i]);
		sha256_step2_host(f, g, h, a, b, c, d, e, in, 3, cpu_K[19 + 16 * i]);
		sha256_step2_host(e, f, g, h, a, b, c, d, in, 4, cpu_K[20 + 16 * i]);
		sha256_step2_host(d, e, f, g, h, a, b, c, in, 5, cpu_K[21 + 16 * i]);
		sha256_step2_host(c, d, e, f, g, h, a, b, in, 6, cpu_K[22 + 16 * i]);
		sha256_step2_host(b, c, d, e, f, g, h, a, in, 7, cpu_K[23 + 16 * i]);
		sha256_step2_host(a, b, c, d, e, f, g, h, in, 8, cpu_K[24 + 16 * i]);
		sha256_step2_host(h, a, b, c, d, e, f, g, in, 9, cpu_K[25 + 16 * i]);
		sha256_step2_host(g, h, a, b, c, d, e, f, in, 10, cpu_K[26 + 16 * i]);
		sha256_step2_host(f, g, h, a, b, c, d, e, in, 11, cpu_K[27 + 16 * i]);
		sha256_step2_host(e, f, g, h, a, b, c, d, in, 12, cpu_K[28 + 16 * i]);
		sha256_step2_host(d, e, f, g, h, a, b, c, in, 13, cpu_K[29 + 16 * i]);
		sha256_step2_host(c, d, e, f, g, h, a, b, in, 14, cpu_K[30 + 16 * i]);
		sha256_step2_host(b, c, d, e, f, g, h, a, in, 15, cpu_K[31 + 16 * i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

//
// Device code
//

#define xor3b(a,b,c) (a ^ b ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x, 2), ROTR32(x, 13), ROTR32(x, 22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x, 6), ROTR32(x, 11), ROTR32(x, 25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x, 7), ROTR32(x, 18), (x >> 3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x, 17), ROTR32(x, 19), (x >> 10));
}

__device__ __forceinline__
static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t in, const uint32_t Kshared)
{
	h += bsg2_1(e) + xandx(e, f, g) + Kshared + in;
	d += h;
	h += bsg2_0(a) + andor32(a, b, c);
}

__device__ __forceinline__
static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	in[pc] += ssg2_1(in[(pc - 2) & 0xF]) + in[(pc - 7) & 0xF] + ssg2_0(in[(pc - 15) & 0xF]);

	sha2_step1(a, b, c, d, e, f, g, h, in[pc], Kshared);
}

__device__ __forceinline__
static void sha256_round_body(uint32_t in[16], uint32_t state[8])
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step1(a, b, c, d, e, f, g, h, in[0], c_K[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[1], c_K[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[2], c_K[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[3], c_K[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[4], c_K[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[5], c_K[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[6], c_K[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[7], c_K[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[8], c_K[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[9], c_K[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[10], c_K[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[11], c_K[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[12], c_K[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[13], c_K[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[14], c_K[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[15], c_K[15]);

#pragma unroll
	for (int i = 0; i < 3; i++)
	{
		sha2_step2(a, b, c, d, e, f, g, h, in, 0, c_K[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, in, 1, c_K[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, in, 2, c_K[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, in, 3, c_K[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, in, 4, c_K[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, in, 5, c_K[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, in, 6, c_K[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, in, 7, c_K[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, in, 8, c_K[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, in, 9, c_K[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, in, 10, c_K[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, in, 11, c_K[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, in, 12, c_K[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, in, 13, c_K[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, in, 14, c_K[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, in, 15, c_K[31 + 16 * i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

#define sha256dev(a) sha256[thread * 8 + (a)]
#define Bdev(a, b) B[((a) * threads + thread) * 16 + (b)]

__launch_bounds__(32, 1)
__global__ void yescrypt_gpu_hash_k0(int threads, uint32_t startNonce, const uint32_t r, const uint32_t p)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	//if (thread < threads)
	{
		uint32_t nonce = startNonce + thread;
		uint32_t in[16];
		uint32_t result[16];
		uint32_t state1[8], state2[8];
		uint32_t passwd[8];

		in[0] = c_data[16]; in[1] = c_data[17]; in[2] = c_data[18]; in[3] = nonce;
		in[5] = in[6] = in[7] = in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
		in[4] = 0x80000000; in[15] = 0x00000280;
		passwd[0] = cpu_h[0]; passwd[1] = cpu_h[1]; passwd[2] = cpu_h[2]; passwd[3] = cpu_h[3];
		passwd[4] = cpu_h[4]; passwd[5] = cpu_h[5]; passwd[6] = cpu_h[6]; passwd[7] = cpu_h[7];
		sha256_round_body(in, passwd);	// length = 80 * 8 = 640 = 0x280

		in[0] = passwd[0] ^ 0x36363636; in[1] = passwd[1] ^ 0x36363636; in[2] = passwd[2] ^ 0x36363636; in[3] = passwd[3] ^ 0x36363636;
		in[4] = passwd[4] ^ 0x36363636; in[5] = passwd[5] ^ 0x36363636; in[6] = passwd[6] ^ 0x36363636; in[7] = passwd[7] ^ 0x36363636;
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x36363636;
		state1[0] = 0x6A09E667; state1[1] = 0xBB67AE85; state1[2] = 0x3C6EF372; state1[3] = 0xA54FF53A;
		state1[4] = 0x510E527F; state1[5] = 0x9B05688C; state1[6] = 0x1F83D9AB; state1[7] = 0x5BE0CD19;
		sha256_round_body(in, state1);	// inner 64byte

		in[0] = passwd[0] ^ 0x5c5c5c5c; in[1] = passwd[1] ^ 0x5c5c5c5c; in[2] = passwd[2] ^ 0x5c5c5c5c; in[3] = passwd[3] ^ 0x5c5c5c5c;
		in[4] = passwd[4] ^ 0x5c5c5c5c; in[5] = passwd[5] ^ 0x5c5c5c5c; in[6] = passwd[6] ^ 0x5c5c5c5c; in[7] = passwd[7] ^ 0x5c5c5c5c;
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x5c5c5c5c;
		state2[0] = 0x6A09E667; state2[1] = 0xBB67AE85; state2[2] = 0x3C6EF372; state2[3] = 0xA54FF53A;
		state2[4] = 0x510E527F; state2[5] = 0x9B05688C; state2[6] = 0x1F83D9AB; state2[7] = 0x5BE0CD19;
		sha256_round_body(in, state2);	// outer 64byte

		in[0] = c_data[0]; in[1] = c_data[1]; in[2] = c_data[2]; in[3] = c_data[3];
		in[4] = c_data[4]; in[5] = c_data[5]; in[6] = c_data[6]; in[7] = c_data[7];
		in[8] = c_data[8]; in[9] = c_data[9]; in[10] = c_data[10]; in[11] = c_data[11];
		in[12] = c_data[12]; in[13] = c_data[13]; in[14] = c_data[14]; in[15] = c_data[15];
		sha256_round_body(in, state1);	// inner 128byte

#pragma unroll
		for (uint32_t i = 0; i < 2 * r*p; i++)
		{
			in[0] = c_data[16]; in[1] = c_data[17]; in[2] = c_data[18]; in[3] = nonce;
			in[4] = i * 2 + 1; in[5] = 0x80000000; in[15] = 0x000004A0;
			in[6] = in[7] = in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
			result[0] = state1[0]; result[1] = state1[1]; result[2] = state1[2]; result[3] = state1[3];
			result[4] = state1[4]; result[5] = state1[5]; result[6] = state1[6]; result[7] = state1[7];
			sha256_round_body(in, result + 0);	// inner length = 148 * 8 = 1184 = 0x4A0

			in[0] = result[0]; in[1] = result[1]; in[2] = result[2]; in[3] = result[3];
			in[4] = result[4]; in[5] = result[5]; in[6] = result[6]; in[7] = result[7];
			in[8] = 0x80000000; in[15] = 0x00000300;
			in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
			result[0] = state2[0]; result[1] = state2[1]; result[2] = state2[2]; result[3] = state2[3];
			result[4] = state2[4]; result[5] = state2[5]; result[6] = state2[6]; result[7] = state2[7];
			sha256_round_body(in, result + 0);	// outer length = 96 * 8 = 768 = 0x300

			in[0] = c_data[16]; in[1] = c_data[17]; in[2] = c_data[18]; in[3] = nonce;
			in[4] = i * 2 + 2; in[5] = 0x80000000; in[15] = 0x000004A0;
			in[6] = in[7] = in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
			result[8] = state1[0]; result[9] = state1[1]; result[10] = state1[2]; result[11] = state1[3];
			result[12] = state1[4]; result[13] = state1[5]; result[14] = state1[6]; result[15] = state1[7];
			sha256_round_body(in, result + 8);	// inner length = 148 * 8 = 1184 = 0x4A0

			in[0] = result[8]; in[1] = result[9]; in[2] = result[10]; in[3] = result[11];
			in[4] = result[12]; in[5] = result[13]; in[6] = result[14]; in[7] = result[15];
			in[8] = 0x80000000; in[15] = 0x00000300;
			in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
			result[8] = state2[0]; result[9] = state2[1]; result[10] = state2[2]; result[11] = state2[3];
			result[12] = state2[4]; result[13] = state2[5]; result[14] = state2[6]; result[15] = state2[7];
			sha256_round_body(in, result + 8);	// outer length = 96 * 8 = 768 = 0x300

			*(uint4*)&Bdev(i, 0) = make_uint4(cuda_swab32(result[0]), cuda_swab32(result[5]), cuda_swab32(result[10]), cuda_swab32(result[15]));
			*(uint4*)&Bdev(i, 4) = make_uint4(cuda_swab32(result[4]), cuda_swab32(result[9]), cuda_swab32(result[14]), cuda_swab32(result[3]));
			*(uint4*)&Bdev(i, 8) = make_uint4(cuda_swab32(result[8]), cuda_swab32(result[13]), cuda_swab32(result[2]), cuda_swab32(result[7]));
			*(uint4*)&Bdev(i, 12) = make_uint4(cuda_swab32(result[12]), cuda_swab32(result[1]), cuda_swab32(result[6]), cuda_swab32(result[11]));

			if (i == 0) {
				sha256dev(0) = result[0];
				sha256dev(1) = result[1];
				sha256dev(2) = result[2];
				sha256dev(3) = result[3];
				sha256dev(4) = result[4];
				sha256dev(5) = result[5];
				sha256dev(6) = result[6];
				sha256dev(7) = result[7];
			}
		}
	}
}

__launch_bounds__(32, 1) 
__global__ void yescrypt_gpu_hash_k0_112bytes(int threads, uint32_t startNonce, const uint32_t r, const uint32_t p) 
{ 
	int thread = (blockDim.x * blockIdx.x + threadIdx.x); 

	//if (thread < threads) 
	{ 
		uint32_t nonce = startNonce + thread; 
		uint32_t in[16]; 
		uint32_t result[16]; 
		uint32_t state1[8], state2[8]; 
		uint32_t passwd[8]; 

		in[0] = c_data[16]; in[1] = c_data[17]; in[2] = c_data[18]; in[3] = nonce; 
		in[4] = c_data[20]; in[5] = c_data[21]; in[6] = c_data[22]; in[7] = c_data[23]; 
		in[8] = c_data[24]; in[9] = c_data[25]; in[10] = c_data[26]; in[11] = c_data[27]; 
		in[12] = 0x80000000; in[13] = in[14] = 0x00000000; in[15] = 0x00000380; 
		passwd[0] = cpu_h[0]; passwd[1] = cpu_h[1]; passwd[2] = cpu_h[2]; passwd[3] = cpu_h[3]; 
		passwd[4] = cpu_h[4]; passwd[5] = cpu_h[5]; passwd[6] = cpu_h[6]; passwd[7] = cpu_h[7]; 
		sha256_round_body(in, passwd);	// length = 112 * 8 = 896 = 0x380 

		in[0] = passwd[0] ^ 0x36363636; in[1] = passwd[1] ^ 0x36363636; in[2] = passwd[2] ^ 0x36363636; in[3] = passwd[3] ^ 0x36363636; 
		in[4] = passwd[4] ^ 0x36363636; in[5] = passwd[5] ^ 0x36363636; in[6] = passwd[6] ^ 0x36363636; in[7] = passwd[7] ^ 0x36363636; 
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x36363636; 
		state1[0] = 0x6A09E667; state1[1] = 0xBB67AE85; state1[2] = 0x3C6EF372; state1[3] = 0xA54FF53A; 
		state1[4] = 0x510E527F; state1[5] = 0x9B05688C; state1[6] = 0x1F83D9AB; state1[7] = 0x5BE0CD19; 
		sha256_round_body(in, state1);	// inner 64byte 

		in[0] = passwd[0] ^ 0x5c5c5c5c; in[1] = passwd[1] ^ 0x5c5c5c5c; in[2] = passwd[2] ^ 0x5c5c5c5c; in[3] = passwd[3] ^ 0x5c5c5c5c; 
		in[4] = passwd[4] ^ 0x5c5c5c5c; in[5] = passwd[5] ^ 0x5c5c5c5c; in[6] = passwd[6] ^ 0x5c5c5c5c; in[7] = passwd[7] ^ 0x5c5c5c5c; 
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x5c5c5c5c; 
		state2[0] = 0x6A09E667; state2[1] = 0xBB67AE85; state2[2] = 0x3C6EF372; state2[3] = 0xA54FF53A; 
		state2[4] = 0x510E527F; state2[5] = 0x9B05688C; state2[6] = 0x1F83D9AB; state2[7] = 0x5BE0CD19; 
		sha256_round_body(in, state2);	// outer 64byte 

		in[0] = c_data[0]; in[1] = c_data[1]; in[2] = c_data[2]; in[3] = c_data[3]; 
		in[4] = c_data[4]; in[5] = c_data[5]; in[6] = c_data[6]; in[7] = c_data[7]; 
		in[8] = c_data[8]; in[9] = c_data[9]; in[10] = c_data[10]; in[11] = c_data[11]; 
		in[12] = c_data[12]; in[13] = c_data[13]; in[14] = c_data[14]; in[15] = c_data[15]; 
		sha256_round_body(in, state1);	// inner 128byte 

#pragma unroll 
		for (uint32_t i = 0; i < 2 * r*p; i++) 
		{ 
			in[0] = c_data[16]; in[1] = c_data[17]; in[2] = c_data[18]; in[3] = nonce; 
			in[4] = c_data[20]; in[5] = c_data[21]; in[6] = c_data[22]; in[7] = c_data[23]; 
			in[8] = c_data[24]; in[9] = c_data[25]; in[10] = c_data[26]; in[11] = c_data[27]; 
			in[12] = i * 2 + 1; in[13] = 0x80000000; in[14] = 0x00000000; in[15] = 0x000005A0; 
			result[0] = state1[0]; result[1] = state1[1]; result[2] = state1[2]; result[3] = state1[3]; 
			result[4] = state1[4]; result[5] = state1[5]; result[6] = state1[6]; result[7] = state1[7]; 
			sha256_round_body(in, result + 0);	// inner length = 180 * 8 = 1184 = 0x5A0 

			in[0] = result[0]; in[1] = result[1]; in[2] = result[2]; in[3] = result[3]; 
			in[4] = result[4]; in[5] = result[5]; in[6] = result[6]; in[7] = result[7]; 
			in[8] = 0x80000000; in[15] = 0x00000300; 
			in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000; 
			result[0] = state2[0]; result[1] = state2[1]; result[2] = state2[2]; result[3] = state2[3]; 
			result[4] = state2[4]; result[5] = state2[5]; result[6] = state2[6]; result[7] = state2[7]; 
			sha256_round_body(in, result + 0);	// outer length = 96 * 8 = 768 = 0x300 

			in[0] = c_data[16]; in[1] = c_data[17]; in[2] = c_data[18]; in[3] = nonce; 
			in[4] = c_data[20]; in[5] = c_data[21]; in[6] = c_data[22]; in[7] = c_data[23];
 			in[8] = c_data[24]; in[9] = c_data[25]; in[10] = c_data[26]; in[11] = c_data[27]; 
			in[12] = i * 2 + 2; in[13] = 0x80000000; in[14] = 0x00000000; in[15] = 0x000005A0; 
			result[8] = state1[0]; result[9] = state1[1]; result[10] = state1[2]; result[11] = state1[3]; 
			result[12] = state1[4]; result[13] = state1[5]; result[14] = state1[6]; result[15] = state1[7]; 
			sha256_round_body(in, result + 8);	// inner length = 180 * 8 = 1184 = 0x5A0 

			in[0] = result[8]; in[1] = result[9]; in[2] = result[10]; in[3] = result[11]; 
			in[4] = result[12]; in[5] = result[13]; in[6] = result[14]; in[7] = result[15]; 
			in[8] = 0x80000000; in[15] = 0x00000300; 
			in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000; 
			result[8] = state2[0]; result[9] = state2[1]; result[10] = state2[2]; result[11] = state2[3]; 
			result[12] = state2[4]; result[13] = state2[5]; result[14] = state2[6]; result[15] = state2[7]; 
			sha256_round_body(in, result + 8);	// outer length = 96 * 8 = 768 = 0x300 

			*(uint4*)&Bdev(i, 0) = make_uint4(cuda_swab32(result[0]), cuda_swab32(result[5]), cuda_swab32(result[10]), cuda_swab32(result[15])); 
			*(uint4*)&Bdev(i, 4) = make_uint4(cuda_swab32(result[4]), cuda_swab32(result[9]), cuda_swab32(result[14]), cuda_swab32(result[3])); 
			*(uint4*)&Bdev(i, 8) = make_uint4(cuda_swab32(result[8]), cuda_swab32(result[13]), cuda_swab32(result[2]), cuda_swab32(result[7])); 
			*(uint4*)&Bdev(i, 12) = make_uint4(cuda_swab32(result[12]), cuda_swab32(result[1]), cuda_swab32(result[6]), cuda_swab32(result[11])); 

			if (i == 0) { 
				sha256dev(0) = result[0]; 
				sha256dev(1) = result[1]; 
				sha256dev(2) = result[2]; 
				sha256dev(3) = result[3]; 
				sha256dev(4) = result[4]; 
				sha256dev(5) = result[5]; 
				sha256dev(6) = result[6]; 
				sha256dev(7) = result[7]; 
			} 
		} 
	} 
} 

__launch_bounds__(32, 1) 
__global__ void yescrypt_gpu_hash_k5(int threads, uint32_t startNonce, uint32_t *nonceVector, uint32_t target, const uint32_t r, const uint32_t p)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	//if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		uint32_t in[16];
		uint32_t buf[8];

		uint32_t state1[8];
		uint32_t state2[8];

		state1[0] = state2[0] = sha256dev(0);
		state1[1] = state2[1] = sha256dev(1);
		state1[2] = state2[2] = sha256dev(2);
		state1[3] = state2[3] = sha256dev(3);
		state1[4] = state2[4] = sha256dev(4);
		state1[5] = state2[5] = sha256dev(5);
		state1[6] = state2[6] = sha256dev(6);
		state1[7] = state2[7] = sha256dev(7);

		in[0] = state1[0] ^ 0x36363636; in[1] = state1[1] ^ 0x36363636; in[2] = state1[2] ^ 0x36363636; in[3] = state1[3] ^ 0x36363636;
		in[4] = state1[4] ^ 0x36363636; in[5] = state1[5] ^ 0x36363636; in[6] = state1[6] ^ 0x36363636; in[7] = state1[7] ^ 0x36363636;
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x36363636;
		state1[0] = 0x6A09E667; state1[1] = 0xBB67AE85; state1[2] = 0x3C6EF372; state1[3] = 0xA54FF53A;
		state1[4] = 0x510E527F; state1[5] = 0x9B05688C; state1[6] = 0x1F83D9AB; state1[7] = 0x5BE0CD19;
		sha256_round_body(in, state1);	// inner 64byte

		in[0] = state2[0] ^ 0x5c5c5c5c; in[1] = state2[1] ^ 0x5c5c5c5c; in[2] = state2[2] ^ 0x5c5c5c5c; in[3] = state2[3] ^ 0x5c5c5c5c;
		in[4] = state2[4] ^ 0x5c5c5c5c; in[5] = state2[5] ^ 0x5c5c5c5c; in[6] = state2[6] ^ 0x5c5c5c5c; in[7] = state2[7] ^ 0x5c5c5c5c;
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x5c5c5c5c;
		state2[0] = 0x6A09E667; state2[1] = 0xBB67AE85; state2[2] = 0x3C6EF372; state2[3] = 0xA54FF53A;
		state2[4] = 0x510E527F; state2[5] = 0x9B05688C; state2[6] = 0x1F83D9AB; state2[7] = 0x5BE0CD19;
		sha256_round_body(in, state2);	// outer 64byte

		for (uint32_t i = 0; i < 2 * r * p; i++)
		{
			in[0] = Bdev(i, 0); in[1] = Bdev(i, 13); in[2] = Bdev(i, 10); in[3] = Bdev(i, 7);
			in[4] = Bdev(i, 4); in[5] = Bdev(i, 1); in[6] = Bdev(i, 14); in[7] = Bdev(i, 11);
			in[8] = Bdev(i, 8); in[9] = Bdev(i, 5); in[10] = Bdev(i, 2); in[11] = Bdev(i, 15);
			in[12] = Bdev(i, 12); in[13] = Bdev(i, 9); in[14] = Bdev(i, 6); in[15] = Bdev(i, 3);
			in[0] = cuda_swab32(in[0]); in[1] = cuda_swab32(in[1]); in[2] = cuda_swab32(in[2]); in[3] = cuda_swab32(in[3]);
			in[4] = cuda_swab32(in[4]); in[5] = cuda_swab32(in[5]); in[6] = cuda_swab32(in[6]); in[7] = cuda_swab32(in[7]);
			in[8] = cuda_swab32(in[8]); in[9] = cuda_swab32(in[9]); in[10] = cuda_swab32(in[10]); in[11] = cuda_swab32(in[11]);
			in[12] = cuda_swab32(in[12]); in[13] = cuda_swab32(in[13]); in[14] = cuda_swab32(in[14]); in[15] = cuda_swab32(in[15]);
			sha256_round_body(in, state1);	// inner 1088byte
		}
		in[0] = 0x00000001; in[1] = 0x80000000; in[15] = (64 + 128 * r * p + 4) * 8;
		in[2] = in[3] = in[4] = in[5] = in[6] = in[7] = in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
		sha256_round_body(in, state1);	// inner length = 1092 * 8 = 8736 = 0x2220

		in[0] = state1[0]; in[1] = state1[1]; in[2] = state1[2]; in[3] = state1[3];
		in[4] = state1[4]; in[5] = state1[5]; in[6] = state1[6]; in[7] = state1[7];
		in[8] = 0x80000000; in[15] = 0x00000300;
		in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
		buf[0] = state2[0]; buf[1] = state2[1]; buf[2] = state2[2]; buf[3] = state2[3];
		buf[4] = state2[4]; buf[5] = state2[5]; buf[6] = state2[6]; buf[7] = state2[7];
		sha256_round_body(in, buf);	// outer length = 96 * 8 = 768 = 0x300

									//hmac and final sha
		in[0] = buf[0] ^ 0x36363636;
		in[1] = buf[1] ^ 0x36363636;
		in[2] = buf[2] ^ 0x36363636;
		in[3] = buf[3] ^ 0x36363636;
		in[4] = buf[4] ^ 0x36363636;
		in[5] = buf[5] ^ 0x36363636;
		in[6] = buf[6] ^ 0x36363636;
		in[7] = buf[7] ^ 0x36363636;
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x36363636;
		state1[0] = state2[0] = 0x6A09E667;
		state1[1] = state2[1] = 0xBB67AE85;
		state1[2] = state2[2] = 0x3C6EF372;
		state1[3] = state2[3] = 0xA54FF53A;
		state1[4] = state2[4] = 0x510E527F;
		state1[5] = state2[5] = 0x9B05688C;
		state1[6] = state2[6] = 0x1F83D9AB;
		state1[7] = state2[7] = 0x5BE0CD19;
		sha256_round_body(in, state1);	// inner 64byte

		in[0] = buf[0] ^ 0x5c5c5c5c;
		in[1] = buf[1] ^ 0x5c5c5c5c;
		in[2] = buf[2] ^ 0x5c5c5c5c;
		in[3] = buf[3] ^ 0x5c5c5c5c;
		in[4] = buf[4] ^ 0x5c5c5c5c;
		in[5] = buf[5] ^ 0x5c5c5c5c;
		in[6] = buf[6] ^ 0x5c5c5c5c;
		in[7] = buf[7] ^ 0x5c5c5c5c;
		in[8] = in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = in[15] = 0x5c5c5c5c;
		sha256_round_body(in, state2);	// outer 64byte

		in[0] = client_key[0]; in[1] = client_key[1]; in[2] = client_key[2]; in[3] = client_key[3];
		in[4] = client_key[4]; in[5] = client_key[5]; in[6] = client_key[6]; in[7] = client_key[7];
		in[8] = client_key[8]; in[9] = client_key[9]; in[10] = client_key[10]; in[11] = client_key[11];
		in[12] = client_key[12]; in[13] = client_key[13]; in[14] = client_key[14]; in[15] = client_key[15];
		sha256_round_body(in, state1);	// inner length = 74 * 8 = 592 = 0x250

		if (client_key_len[0] >= 56 || client_key_len[0] == 0) {
			in[0] = client_key[16]; in[1] = client_key[17]; in[2] = client_key[18];
			if (client_key_len[0] == 0) in[3] = nonce;
			else in[3] = client_key[19];
			in[4] = client_key[20]; in[5] = client_key[21]; in[6] = client_key[22]; in[7] = client_key[23];
			in[8] = client_key[24]; in[9] = client_key[25]; in[10] = client_key[26]; in[11] = client_key[27];
			in[12] = client_key[28]; in[13] = client_key[29]; in[14] = client_key[30]; in[15] = client_key[31];
			sha256_round_body(in, state1);	// inner length = 74 * 8 = 592 = 0x250
		}

		in[0] = state1[0]; in[1] = state1[1]; in[2] = state1[2]; in[3] = state1[3];
		in[4] = state1[4]; in[5] = state1[5]; in[6] = state1[6]; in[7] = state1[7];
		in[8] = 0x80000000; in[15] = 0x00000300;
		in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
		sha256_round_body(in, state2);	// outer length = 96 * 8 = 768 = 0x300

		in[0] = state2[0]; in[1] = state2[1]; in[2] = state2[2]; in[3] = state2[3];
		in[4] = state2[4]; in[5] = state2[5]; in[6] = state2[6]; in[7] = state2[7];
		in[8] = 0x80000000; in[15] = 0x00000100;
		in[9] = in[10] = in[11] = in[12] = in[13] = in[14] = 0x00000000;
		buf[0] = 0x6A09E667; buf[1] = 0xBB67AE85; buf[2] = 0x3C6EF372; buf[3] = 0xA54FF53A;
		buf[4] = 0x510E527F; buf[5] = 0x9B05688C; buf[6] = 0x1F83D9AB; buf[7] = 0x5BE0CD19;
		sha256_round_body(in, buf);	// length = 32 * 8 = 256 = 0x100

		if (cuda_swab32(buf[7]) <= target)
		{
			uint32_t tmp = atomicExch(&nonceVector[0], nonce);
			if (tmp != 0)
				nonceVector[1] = tmp;
		}
	}
}

#undef sha256dev
#undef Bdev

//////////////////////////////// end sha256 mechanism ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

#define SALSA(a,b,c,d) { \
    b^=ROTL(a+d,  7);    \
    c^=ROTL(b+a,  9);    \
    d^=ROTL(c+b, 13);    \
    a^=ROTL(d+c, 18);     \
}

#define SALSA_CORE(x0, x1, x2, x3) { \
	uint32_t t0, t1, t2, t3; \
	t0 = x0; t1 = x1; t2 = x2; t3 = x3; \
	for (int idx = 0; idx < 4; ++idx) { \
		SALSA(x0, x1, x2, x3); \
		WarpShuffle3(x1,x2,x3,x1,x2,x3,threadIdx.x + 3,threadIdx.x + 2,threadIdx.x + 1,4);\
		SALSA(x0, x3, x2, x1); \
		WarpShuffle3(x1,x2,x3,x1,x2,x3,threadIdx.x + 1,threadIdx.x + 2,threadIdx.x + 3,4);\
	} \
	x0 += t0; x1 += t1; x2 += t2; x3 += t3; \
}
/**
* p2floor(x):
* Largest power of 2 not greater than argument.
*/
__device__ __forceinline__ uint32_t p2floor(uint32_t x)
{
	uint32_t y;
	while (y = x & (x - 1))
		x = y;
	return x;
}

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

static __device__ __forceinline__ uint32_t __ldL1(const uint32_t *ptr)
{
	uint32_t ret;
	asm("ld.global.ca.u32 %0, [%1];" : "=r"(ret) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint32_t *ptr, const uint32_t value)
{
	asm("st.global.wb.u32 [%0+0], %1;" ::__LDG_PTR(ptr), "r"(value));
}

__device__ __forceinline__ uint2 mad64(const uint32_t a, const uint32_t b, uint2 c)
{
#if 0
	return vectorize((uint64_t)a * (uint64_t)b) + c;
#else
	asm("{\n\t"
		"mad.lo.cc.u32 %0, %2, %3, %0; \n\t"
		"madc.hi.u32 %1, %2, %3, %1; \n\t"
		"}\n\t" : "+r"(c.x), "+r"(c.y) : "r"(a), "r"(b)
	);
	return c;
#endif
}

#define Bdev(a, b) B[((a) * threads + thread) * 16 + (b) * 4 + threadIdx.x]
#define Sdev(a, b) S[(thread_part_4 * 128 + (a)) * 16 + (b) * 4 + threadIdx.x]

__global__ __launch_bounds__(32, 1)
void yescrypt_gpu_hash_k1(int threads, uint32_t startNonce, uint32_t offset)
{
	uint32_t thread_part_4 = (8 * blockIdx.x + threadIdx.y);
	uint32_t thread = thread_part_4 + offset;

	//if (thread < threads)
	{
		uint32_t n, i, j;
		uint32_t x[8];

		x[0] = Bdev(0, 0);
		x[1] = Bdev(0, 1);
		x[2] = Bdev(0, 2);
		x[3] = Bdev(0, 3);
		x[4] = Bdev(1, 0);
		x[5] = Bdev(1, 1);
		x[6] = Bdev(1, 2);
		x[7] = Bdev(1, 3);

		for (n = 1, i = 0; i < 64; i++)
		{
			/* 3: V_i <-- X */
			__stL1(&Sdev(i * 2 + 0, 0), x[0]);
			__stL1(&Sdev(i * 2 + 0, 1), x[1]);
			__stL1(&Sdev(i * 2 + 0, 2), x[2]);
			__stL1(&Sdev(i * 2 + 0, 3), x[3]);
			__stL1(&Sdev(i * 2 + 1, 0), x[4]);
			__stL1(&Sdev(i * 2 + 1, 1), x[5]);
			__stL1(&Sdev(i * 2 + 1, 2), x[6]);
			__stL1(&Sdev(i * 2 + 1, 3), x[7]);

			if (i > 1) {
				if ((i & (i - 1)) == 0) n = i;
				j = WarpShuffle(x[4], 0, 4) & (n - 1);
				j += i - n;

				x[0] ^= __ldL1(&Sdev(j * 2 + 0, 0));
				x[1] ^= __ldL1(&Sdev(j * 2 + 0, 1));
				x[2] ^= __ldL1(&Sdev(j * 2 + 0, 2));
				x[3] ^= __ldL1(&Sdev(j * 2 + 0, 3));
				x[4] ^= __ldL1(&Sdev(j * 2 + 1, 0));
				x[5] ^= __ldL1(&Sdev(j * 2 + 1, 1));
				x[6] ^= __ldL1(&Sdev(j * 2 + 1, 2));
				x[7] ^= __ldL1(&Sdev(j * 2 + 1, 3));
			}

			x[0] ^= x[4]; x[1] ^= x[5]; x[2] ^= x[6]; x[3] ^= x[7];
			SALSA_CORE(x[0], x[1], x[2], x[3]);
			x[4] ^= x[0]; x[5] ^= x[1]; x[6] ^= x[2]; x[7] ^= x[3];
			SALSA_CORE(x[4], x[5], x[6], x[7]);
		}

		Bdev(0, 0) = x[0];
		Bdev(0, 1) = x[1];
		Bdev(0, 2) = x[2];
		Bdev(0, 3) = x[3];
		Bdev(1, 0) = x[4];
		Bdev(1, 1) = x[5];
		Bdev(1, 2) = x[6];
		Bdev(1, 3) = x[7];
	}
}

#undef Bdev
#undef Sdev

#define Vdev(a, b) v[((a) * 16 + (b)) * 32]
#define Bdev(a) B[((a) * threads + thread) * 16 + threadIdx.x]
#define Sdev(a) S[(thread_part_4 * 128 + (a)) * 16 + threadIdx.x]
#define Shared(a) *(uint2*)&shared_mem[(threadIdx.y * 512 + (a)) * 4 + (threadIdx.x & 2)]

__global__ __launch_bounds__(32, 1) void yescrypt_gpu_hash_k2c_r8(int threads, uint32_t startNonce, uint32_t offset1, uint32_t offset2, uint32_t start, uint32_t end, const uint32_t N)
{
	uint32_t thread_part_16 = (2 * blockIdx.x + threadIdx.y);
	uint32_t thread_part_4 = thread_part_16 + offset1;
	uint32_t thread = thread_part_16 + offset2;
	extern __shared__ uint32_t shared_mem[];

	const uint32_t r = 8;

	uint32_t *v = &V[blockIdx.x * N * r * 2 * 32 + threadIdx.y * 16 + threadIdx.x];

	//if (thread < threads)
	{
		uint32_t n, i, j, k;
		uint32_t x0, x1, x2, x3;
		uint2 buf;
		uint32_t x[r * 2];

		for (k = 0; k < 128; k++)
			shared_mem[(threadIdx.y * 128 + k) * 16 + threadIdx.x] = Sdev(k);

#pragma unroll
		for (k = 0; k < r * 2; k++)
			x[k] = Bdev(k);

		for (n = p2floor(start), i = start; i < end; i++) {
#pragma unroll
			for (k = 0; k < r * 2; k++) {
				x3 = x[k];
				__stL1(&Vdev(i, k), x3);
			}

			if (i > 1) {
				if ((i & (i - 1)) == 0) n = i;
				j = WarpShuffle(x3, 0, 16) & (n - 1);
				j += i - n;

				for (k = 0; k < r * 2; k++) {
					x3 = x[k] ^ __ldL1(&Vdev(j, k));
					x[k] = x3;
				}
			}

#pragma unroll
			for (k = 0; k < r * 2; k++) {
				x3 ^= x[k];
				WarpShuffle2(buf.x, buf.y, x3, x3, 0, 1, 2);
#pragma unroll
				for (j = 0; j < 6; j++) {
					WarpShuffle2(x0, x1, buf.x, buf.y, 0, 0, 4);
					x0 = ((x0 >> 4) & 255) + 0;
					x1 = ((x1 >> 4) & 255) + 256;
					buf = mad64(buf.x, buf.y, Shared(x0));
					buf ^= Shared(x1);
				}
				if (threadIdx.x & 1) x3 = buf.y;
				else x3 = buf.x;

				x[k] = x3;
			}
			WarpShuffle4(x0, x1, x2, x3, x3, x3, x3, x3, 0 + (threadIdx.x & 3), 4 + (threadIdx.x & 3), 8 + (threadIdx.x & 3), 12 + (threadIdx.x & 3), 16);
			SALSA_CORE(x0, x1, x2, x3);
			if (threadIdx.x < 4) x3 = x0;
			else if (threadIdx.x < 8) x3 = x1;
			else if (threadIdx.x < 12) x3 = x2;

			x[r * 2 - 1] = x3;
		}

#pragma unroll
		for (k = 0; k < r * 2; k++)
			Bdev(k) = x[k];

	}
}

__global__ __launch_bounds__(32, 1) void yescrypt_gpu_hash_k2c1_r8(int threads, uint32_t startNonce, uint32_t offset1, uint32_t offset2, uint32_t start, uint32_t end, const uint32_t N)
{
	uint32_t thread_part_16 = (2 * blockIdx.x + threadIdx.y);
	uint32_t thread_part_4 = thread_part_16 + offset1;
	uint32_t thread = thread_part_16 + offset2;
	extern __shared__ uint32_t shared_mem[];

	const uint32_t r = 8;

	uint32_t *v = &V[blockIdx.x * N * r * 2 * 32 + threadIdx.y * 16 + threadIdx.x];

	//if (thread < threads)
	{
		uint32_t j, k;
		uint32_t x0, x1, x2, x3;
		uint2 buf;
		uint32_t x[r * 2];

		for (k = 0; k < 128; k++)
			shared_mem[(threadIdx.y * 128 + k) * 16 + threadIdx.x] = Sdev(k);

#pragma unroll
		for (k = 0; k < r * 2; k++)
			x[k] = Bdev(k);

		for (uint32_t z = start; z < end; z++)
		{
			j = WarpShuffle(x[r * 2 - 1], 0, 16) & (N - 1);

#pragma unroll
			for (k = 0; k < r * 2; k++)
				x[k] ^= __ldL1(&Vdev(j, k));

#pragma unroll
			for (k = 0; k < r * 2; k++) {
				x3 = x[k];
				__stL1(&Vdev(j, k), x3);
			}

#pragma unroll
			for (k = 0; k < r * 2; k++) {
				x3 ^= x[k];
				WarpShuffle2(buf.x, buf.y, x3, x3, 0, 1, 2);
#pragma unroll
				for (j = 0; j < 6; j++) {
					WarpShuffle2(x0, x1, buf.x, buf.y, 0, 0, 4);
					x0 = ((x0 >> 4) & 255) + 0;
					x1 = ((x1 >> 4) & 255) + 256;
					buf = mad64(buf.x, buf.y, Shared(x0));
					buf ^= Shared(x1);
				}
				if (threadIdx.x & 1) x3 = buf.y;
				else x3 = buf.x;

				x[k] = x3;
			}
			WarpShuffle4(x0, x1, x2, x3, x3, x3, x3, x3, 0 + (threadIdx.x & 3), 4 + (threadIdx.x & 3), 8 + (threadIdx.x & 3), 12 + (threadIdx.x & 3), 16);
			SALSA_CORE(x0, x1, x2, x3);
			if (threadIdx.x < 4) x3 = x0;
			else if (threadIdx.x < 8) x3 = x1;
			else if (threadIdx.x < 12) x3 = x2;

			x[r * 2 - 1] = x3;
		}

#pragma unroll
		for (k = 0; k < r * 2; k++)
			Bdev(k) = x[k];
	}
}

#undef Vdev
#undef Bdev
#undef Sdev

#define Vdev(a, b) v[((a) * r*2 + (b)) * 32]
#define Bdev(a) B[((a) * threads + thread) * 16 + threadIdx.x]
#define Sdev(a) S[(thread_part_4 * 128 + (a)) * 16 + threadIdx.x]

__global__ __launch_bounds__(32, 1) void yescrypt_gpu_hash_k2c(int threads, uint32_t startNonce, uint32_t offset1, uint32_t offset2, uint32_t start, uint32_t end, const uint32_t N, const uint32_t r, const uint32_t p)
{
	uint32_t thread_part_16 = (2 * blockIdx.x + threadIdx.y);
	uint32_t thread_part_4 = thread_part_16 + offset1;
	uint32_t thread = thread_part_16 + offset2;
	extern __shared__ uint32_t shared_mem[];

	uint32_t *v = &V[blockIdx.x * N * r * 2 * 32 + threadIdx.y * 16 + threadIdx.x];

	//if (thread < threads)
	{
		uint32_t n, i, j, k;
		uint32_t x0, x1, x2, x3;
		uint2 buf;
		uint32_t x[256];

		for (k = 0; k < 128; k++)
			shared_mem[(threadIdx.y * 128 + k) * 16 + threadIdx.x] = Sdev(k);

		for (k = 0; k < r * 2; k++) {
			x3 = Bdev(k);
			x[k] = x3;
		}

		for (n = p2floor(start), i = start; i < end; i++) {

			for (k = 0; k < r * 2; k++) {
				x3 = x[k];
				__stL1(&Vdev(i, k), x3);
			}

			if (i > 1) {
				if ((i & (i - 1)) == 0) n = i;
				j = WarpShuffle(x3, 0, 16) & (n - 1);
				j += i - n;

				for (k = 0; k < r * 2; k++) {
					x3 = x[k] ^ __ldL1(&Vdev(j, k));
					x[k] = x3;
				}
			}

			for (k = 0; k < r * 2; k++) {
				x3 ^= x[k];
				WarpShuffle2(buf.x, buf.y, x3, x3, 0, 1, 2);
#pragma unroll
				for (j = 0; j < 6; j++) {
					WarpShuffle2(x0, x1, buf.x, buf.y, 0, 0, 4);
					x0 = ((x0 >> 4) & 255) + 0;
					x1 = ((x1 >> 4) & 255) + 256;
					buf = mad64(buf.x, buf.y, Shared(x0));
					buf ^= Shared(x1);
				}
				if (threadIdx.x & 1) x3 = buf.y;
				else x3 = buf.x;

				x[k] = x3;
			}
			WarpShuffle4(x0, x1, x2, x3, x3, x3, x3, x3, 0 + (threadIdx.x & 3), 4 + (threadIdx.x & 3), 8 + (threadIdx.x & 3), 12 + (threadIdx.x & 3), 16);
			SALSA_CORE(x0, x1, x2, x3);
			if (threadIdx.x < 4) x3 = x0;
			else if (threadIdx.x < 8) x3 = x1;
			else if (threadIdx.x < 12) x3 = x2;

			x[r * 2 - 1] = x3;
		}

		for (k = 0; k < r * 2; k++)
			Bdev(k) = x[k];

	}
}

__global__ __launch_bounds__(32, 1) void yescrypt_gpu_hash_k2c1(int threads, uint32_t startNonce, uint32_t offset1, uint32_t offset2, uint32_t start, uint32_t end, const uint32_t N, const uint32_t r, const uint32_t p)
{
	uint32_t thread_part_16 = (2 * blockIdx.x + threadIdx.y);
	uint32_t thread_part_4 = thread_part_16 + offset1;
	uint32_t thread = thread_part_16 + offset2;
	extern __shared__ uint32_t shared_mem[];

	uint32_t *v = &V[blockIdx.x * N * r * 2 * 32 + threadIdx.y * 16 + threadIdx.x];

	//if (thread < threads)
	{
		uint32_t j, k;
		uint32_t x0, x1, x2, x3;
		uint2 buf;
		uint32_t x[256];

		for (k = 0; k < 128; k++)
			shared_mem[(threadIdx.y * 128 + k) * 16 + threadIdx.x] = Sdev(k);

		for (k = 0; k < r * 2; k++) {
			x3 = Bdev(k);
			x[k] = x3;
		}

		for (uint32_t z = start; z < end; z++)
		{
			j = WarpShuffle(x3, 0, 16) & (N - 1);

			for (k = 0; k < r * 2; k++)
				x[k] ^= __ldL1(&Vdev(j, k));

			for (k = 0; k < r * 2; k++) {
				x3 = x[k];
				__stL1(&Vdev(j, k), x3);
			}

			for (k = 0; k < r * 2; k++) {
				x3 ^= x[k];
				WarpShuffle2(buf.x, buf.y, x3, x3, 0, 1, 2);
#pragma unroll
				for (j = 0; j < 6; j++) {
					WarpShuffle2(x0, x1, buf.x, buf.y, 0, 0, 4);
					x0 = ((x0 >> 4) & 255) + 0;
					x1 = ((x1 >> 4) & 255) + 256;
					buf = mad64(buf.x, buf.y, Shared(x0));
					buf ^= Shared(x1);
				}
				if (threadIdx.x & 1) x3 = buf.y;
				else x3 = buf.x;

				x[k] = x3;
			}
			WarpShuffle4(x0, x1, x2, x3, x3, x3, x3, x3, 0 + (threadIdx.x & 3), 4 + (threadIdx.x & 3), 8 + (threadIdx.x & 3), 12 + (threadIdx.x & 3), 16);
			SALSA_CORE(x0, x1, x2, x3);
			if (threadIdx.x < 4) x3 = x0;
			else if (threadIdx.x < 8) x3 = x1;
			else if (threadIdx.x < 12) x3 = x2;
			x[r * 2 - 1] = x3;
		}

		for (k = 0; k < r * 2; k++)
			Bdev(k) = x[k];
	}
}

__global__ __launch_bounds__(32, 1) void yescrypt_gpu_hash_k2c2(int threads, uint32_t startNonce, uint32_t offset1, uint32_t offset2, const uint32_t N, const uint32_t r, const uint32_t p)
{
	uint32_t thread_part_16 = (2 * blockIdx.x + threadIdx.y);
	uint32_t thread_part_4 = thread_part_16 + offset1;
	uint32_t thread = thread_part_16 + offset2;
	extern __shared__ uint32_t shared_mem[];

	uint32_t *v = &V[blockIdx.x * N * r * 2 * 32 + threadIdx.y * 16 + threadIdx.x];

	//if (thread < threads)
	{
		uint32_t j, k;
		uint32_t x0, x1, x2, x3;
		uint2 buf;
		uint32_t x[256];

		for (k = 0; k < 128; k++)
			shared_mem[(threadIdx.y * 128 + k) * 16 + threadIdx.x] = Sdev(k);

		for (k = 0; k < r * 2; k++) {
			x3 = Bdev(k);
			x[k] = x3;
		}

		for (uint32_t z = 0; z < 2; z++)
		{
			j = WarpShuffle(x3, 0, 16) & (N - 1);

			for (k = 0; k < r * 2; k++) {
				x3 = x[k] ^ __ldL1(&Vdev(j, k));
				x[k] = x3;
			}

			for (k = 0; k < r * 2; k++) {
				x3 ^= x[k];
				WarpShuffle2(buf.x, buf.y, x3, x3, 0, 1, 2);
#pragma unroll
				for (j = 0; j < 6; j++) {
					WarpShuffle2(x0, x1, buf.x, buf.y, 0, 0, 4);
					x0 = ((x0 >> 4) & 255) + 0;
					x1 = ((x1 >> 4) & 255) + 256;
					buf = mad64(buf.x, buf.y, Shared(x0));
					buf ^= Shared(x1);
				}
				if (threadIdx.x & 1) x3 = buf.y;
				else x3 = buf.x;

				x[k] = x3;
			}
			WarpShuffle4(x0, x1, x2, x3, x3, x3, x3, x3, 0 + (threadIdx.x & 3), 4 + (threadIdx.x & 3), 8 + (threadIdx.x & 3), 12 + (threadIdx.x & 3), 16);
			SALSA_CORE(x0, x1, x2, x3);
			if (threadIdx.x < 4) x3 = x0;
			else if (threadIdx.x < 8) x3 = x1;
			else if (threadIdx.x < 12) x3 = x2;
			x[r * 2 - 1] = x3;
		}

		for (k = 0; k < r * 2; k++)
			Bdev(k) = x[k];
	}
}

#undef Vdev
#undef Bdev
#undef Sdev
#undef Shared

__host__
void yescrypt_cpu_init(int thr_id, int threads, uint32_t *d_hash1, uint32_t *d_hash2, uint32_t *d_hash3, uint32_t *d_hash4)
{

	cudaMemcpyToSymbol(B, &d_hash1, sizeof(d_hash1), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(S, &d_hash2, sizeof(d_hash2), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(V, &d_hash3, sizeof(d_hash3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(sha256, &d_hash4, sizeof(d_hash4), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaMalloc(&d_GNonce[thr_id], 2 * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMallocHost(&d_gnounce[thr_id], 2 * sizeof(uint32_t)));
}

__host__
void yescrypt_setTarget(int thr_id, uint32_t pdata[20], char *key, uint32_t key_len, const int perslen)
{
	uint32_t h[8], data[32];

	h[0] = 0x6A09E667; h[1] = 0xBB67AE85; h[2] = 0x3C6EF372; h[3] = 0xA54FF53A;
	h[4] = 0x510E527F; h[5] = 0x9B05688C; h[6] = 0x1F83D9AB; h[7] = 0x5BE0CD19;
	data[0] = ((uint32_t*)pdata)[0]; data[1] = ((uint32_t*)pdata)[1];
	data[2] = ((uint32_t*)pdata)[2]; data[3] = ((uint32_t*)pdata)[3];
	data[4] = ((uint32_t*)pdata)[4]; data[5] = ((uint32_t*)pdata)[5];
	data[6] = ((uint32_t*)pdata)[6]; data[7] = ((uint32_t*)pdata)[7];
	data[8] = ((uint32_t*)pdata)[8]; data[9] = ((uint32_t*)pdata)[9];
	data[10] = ((uint32_t*)pdata)[10]; data[11] = ((uint32_t*)pdata)[11];
	data[12] = ((uint32_t*)pdata)[12]; data[13] = ((uint32_t*)pdata)[13];
	data[14] = ((uint32_t*)pdata)[14]; data[15] = ((uint32_t*)pdata)[15];
	sha256_round_body_host(data, h, cpu_K);

	cudaMemcpyToSymbol(cpu_h, h, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, pdata, 28 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);

	if (key)
	{
		memcpy(data, key, key_len);
		((uint8_t*)data)[key_len] = (uint8_t)0x80;
		memset((uint8_t*)data + key_len + 1, 0, 123 - key_len);
		for (uint32_t i = 0; i < 31; i++)
			data[i] = (data[i] >> 24) | ((data[i] >> 8) & 0xFF00) | ((data[i] & 0xFF00) << 8) | (data[i] << 24);

		if (key_len < 56) data[15] = (key_len + 64) * 8;
		else data[31] = (key_len + 64) * 8;
	}
	else
	{
		if (perslen == 80) { 
			memcpy(data, pdata, 80); 
			data[20] = 0x80000000; 
			data[21] = data[22] = data[23] = data[24] = data[25] = data[26] = data[27] = data[28] = data[29] = data[30] = 0; 
			data[31] = (80 + 64) * 8; 
		} else { 
			memcpy(data, pdata, 112); 
			data[28] = 0x80000000; 
			data[29] = data[30] = 0; 
			data[31] = (112 + 64) * 8; 
		} 
		key_len = 0;
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(client_key, data, 32 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(client_key_len, &key_len, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

__host__ void yescrypt_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resultnonces, uint32_t target, const uint32_t N, const uint32_t r, const uint32_t p, const int is112) 
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	CUDA_SAFE_CALL(cudaMemset(d_GNonce[thr_id], 0, 2 * sizeof(uint32_t)));

	const uint32_t tpb = 32U;
	uint32_t sm = 0;

	dim3 grid(threads / tpb);
	dim3 block(tpb);
	dim3 block2(4U, tpb >> 2);
	dim3 block3(16U, tpb >> 4);

	if (device_sm[dev_id] < 500) {
		cudaFuncSetCacheConfig(yescrypt_gpu_hash_k2c, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(yescrypt_gpu_hash_k2c1, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(yescrypt_gpu_hash_k2c2, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(yescrypt_gpu_hash_k2c_r8, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(yescrypt_gpu_hash_k2c1_r8, cudaFuncCachePreferShared);
	}
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
	else if (device_sm[dev_id] >= 700) {
		cudaFuncSetAttribute(yescrypt_gpu_hash_k2c, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(yescrypt_gpu_hash_k2c1, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(yescrypt_gpu_hash_k2c2, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(yescrypt_gpu_hash_k2c_r8, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaFuncSetAttribute(yescrypt_gpu_hash_k2c1_r8, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
	}
#endif
	if (device_sm[dev_id] < 300) sm = 4 * sizeof(uint32_t) * tpb;

	const uint32_t Nw = ((N + 2) / 3) & ~1;
	const uint32_t Nr = ((N + 2) / 3 + 1) & ~1;

	// 応答が遅いとErrorになるので、loop_countで適度に分割します。
	uint32_t loop_count;
	if (device_sm[dev_id] > 500) loop_count = max(N * r / 16384, 1);
	else if (device_sm[dev_id] == 500) loop_count = max(N * r / 8192, 1);
	else if (device_sm[dev_id] > 300) loop_count = max(N * r / 4096, 1);
	else loop_count = max(N * r / 2048, 1);

	if (is112) 
		yescrypt_gpu_hash_k0_112bytes << <grid, block >> > (threads, startNounce, r, p); 
	else 
		yescrypt_gpu_hash_k0 << <grid, block >> > (threads, startNounce, r, p); 
	CUDA_SAFE_CALL(cudaGetLastError());
	for (uint32_t l = 0; l < p; l++)
	{
		// メモリの使用量を抑えるため、16分割で演算を行います。
		// しかし、yescrypt_gpu_hash_k1で16分割は不向きで、この部分のメモリ使用量も少ないため、
		// yescrypt_gpu_hash_k1は4分割で演算を行います。
		for (uint32_t i = 0; i < 4; i++)
		{
			yescrypt_gpu_hash_k1 << <grid, block2, sm >> > (threads, startNounce, i * (threads >> 2) + l * r * 2 * threads);
			CUDA_SAFE_CALL(cudaGetLastError());
			for (uint32_t j = 0; j < 4; j++)
			{
				if (r == 8) {
					for (uint32_t k = 0; k < loop_count - 1; k++)
						yescrypt_gpu_hash_k2c_r8 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, (N / loop_count) * k, (N / loop_count) * (k + 1), N);
					yescrypt_gpu_hash_k2c_r8 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, N / loop_count * (loop_count - 1), N, N);
					CUDA_SAFE_CALL(cudaGetLastError());
					for (uint32_t k = 0; k < loop_count - 1; k++)
						yescrypt_gpu_hash_k2c1_r8 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, (Nw / loop_count) * k, (Nw / loop_count)* (k + 1), N);
					yescrypt_gpu_hash_k2c1_r8 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, (Nw / loop_count) * (loop_count - 1), Nw, N);
					CUDA_SAFE_CALL(cudaGetLastError());
				}
				else {
					for (uint32_t k = 0; k < loop_count - 1; k++)
						yescrypt_gpu_hash_k2c << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, (N / loop_count) * k, (N / loop_count) * (k + 1), N, r, p);
					yescrypt_gpu_hash_k2c << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, N / loop_count * (loop_count - 1), N, N, r, p);
					CUDA_SAFE_CALL(cudaGetLastError());
					for (uint32_t k = 0; k < loop_count - 1; k++)
						yescrypt_gpu_hash_k2c1 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, (Nw / loop_count) * k, (Nw / loop_count)* (k + 1), N, r, p);
					yescrypt_gpu_hash_k2c1 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, (Nw / loop_count) * (loop_count - 1), Nw, N, r, p);
					CUDA_SAFE_CALL(cudaGetLastError());
				}
				if (Nr - Nw > 0)
					yescrypt_gpu_hash_k2c2 << <grid, block3, 16384 >> > (threads, startNounce, j * (threads >> 4), (i * 4 + j) * (threads >> 4) + l * r * 2 * threads, N, r, p);
				CUDA_SAFE_CALL(cudaGetLastError());
			}
		}
	}
	yescrypt_gpu_hash_k5 << <grid, block >> > (threads, startNounce, d_GNonce[thr_id], target, r, p);
	CUDA_SAFE_CALL(cudaGetLastError());
	if (opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaMemcpy(d_gnounce[thr_id], d_GNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	resultnonces[0] = *(d_gnounce[thr_id]);
	resultnonces[1] = *(d_gnounce[thr_id] + 1);
}

__host__ void yescrypt_cpu_free(int thr_id)
{
	cudaFreeHost(d_gnounce[thr_id]);
	cudaFree(d_GNonce[thr_id]);
}
