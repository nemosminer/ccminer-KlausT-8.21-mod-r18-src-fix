#include "cuda_helper.h"

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */

#define ROTATEUPWARDS7(a)  ROTL32(a,7)
#define ROTATEUPWARDS11(a) ROTL32(a,11)

// Max TPB 1024 (Fermi or later)
#define TPB60 1024
#define TPB52 1024
#define TPB50 1024
#define TPB30 192
#define TPB20 96

// Max BPM 32 (Maxwell or later)
// Max BPM 16 (Kepler)
// Max BPM 8 (Fermi or older)
#define BPM60 2
#define BPM52 1
#define BPM50 1
#define BPM30 4
#define BPM20 6

#if __CUDA_ARCH__ >= 750
#define TPB TPB60
#define BPM BPM60
#elif __CUDA_ARCH__ > 600
#define TPB TPB52
#define BPM BPM52
#elif __CUDA_ARCH__ == 600
#define TPB TPB60
#define BPM BPM60
#elif __CUDA_ARCH__ >= 520
#define TPB TPB52
#define BPM BPM52
#elif __CUDA_ARCH__ >= 500
#define TPB TPB50
#define BPM BPM50
#elif __CUDA_ARCH__ >= 300
#define TPB TPB30
#define BPM BPM30
#else
#define TPB TPB20
#define BPM BPM20
#endif

static __device__ __forceinline__ void rrounds(uint32_t x[32])
{
	int r;

	for (r = 0; r < CUBEHASH_ROUNDS; r += 2)
	{
		/* "add x_0jklm into x_1jklm modulo 2^32" */
		x[16] += x[0];
		x[17] += x[1];
		x[18] += x[2];
		x[19] += x[3];
		x[20] += x[4];
		x[21] += x[5];
		x[22] += x[6];
		x[23] += x[7];
		x[24] += x[8];
		x[25] += x[9];
		x[26] += x[10];
		x[27] += x[11];
		x[28] += x[12];
		x[29] += x[13];
		x[30] += x[14];
		x[31] += x[15];

		/* "rotate x_0jklm upwards by 7 bits" */
		/* "xor x_1~jklm into x_0jklm" */
		x[0] = ROTATEUPWARDS7(x[0]) ^ x[24];
		x[1] = ROTATEUPWARDS7(x[1]) ^ x[25];
		x[2] = ROTATEUPWARDS7(x[2]) ^ x[26];
		x[3] = ROTATEUPWARDS7(x[3]) ^ x[27];
		x[4] = ROTATEUPWARDS7(x[4]) ^ x[28];
		x[5] = ROTATEUPWARDS7(x[5]) ^ x[29];
		x[6] = ROTATEUPWARDS7(x[6]) ^ x[30];
		x[7] = ROTATEUPWARDS7(x[7]) ^ x[31];
		x[8] = ROTATEUPWARDS7(x[8]) ^ x[16];
		x[9] = ROTATEUPWARDS7(x[9]) ^ x[17];
		x[10] = ROTATEUPWARDS7(x[10]) ^ x[18];
		x[11] = ROTATEUPWARDS7(x[11]) ^ x[19];
		x[12] = ROTATEUPWARDS7(x[12]) ^ x[20];
		x[13] = ROTATEUPWARDS7(x[13]) ^ x[21];
		x[14] = ROTATEUPWARDS7(x[14]) ^ x[22];
		x[15] = ROTATEUPWARDS7(x[15]) ^ x[23];

		/* "add x_0jklm into x_1~jk~lm modulo 2^32" */
		x[26] += x[0];
		x[27] += x[1];
		x[24] += x[2];
		x[25] += x[3];
		x[30] += x[4];
		x[31] += x[5];
		x[28] += x[6];
		x[29] += x[7];
		x[18] += x[8];
		x[19] += x[9];
		x[16] += x[10];
		x[17] += x[11];
		x[22] += x[12];
		x[23] += x[13];
		x[20] += x[14];
		x[21] += x[15];

		/* "rotate x_0jklm upwards by 11 bits" */
		/* "xor x_1~j~k~lm into x_0jklm" */
		x[0] = ROTATEUPWARDS11(x[0]) ^ x[30];
		x[1] = ROTATEUPWARDS11(x[1]) ^ x[31];
		x[2] = ROTATEUPWARDS11(x[2]) ^ x[28];
		x[3] = ROTATEUPWARDS11(x[3]) ^ x[29];
		x[4] = ROTATEUPWARDS11(x[4]) ^ x[26];
		x[5] = ROTATEUPWARDS11(x[5]) ^ x[27];
		x[6] = ROTATEUPWARDS11(x[6]) ^ x[24];
		x[7] = ROTATEUPWARDS11(x[7]) ^ x[25];
		x[8] = ROTATEUPWARDS11(x[8]) ^ x[22];
		x[9] = ROTATEUPWARDS11(x[9]) ^ x[23];
		x[10] = ROTATEUPWARDS11(x[10]) ^ x[20];
		x[11] = ROTATEUPWARDS11(x[11]) ^ x[21];
		x[12] = ROTATEUPWARDS11(x[12]) ^ x[18];
		x[13] = ROTATEUPWARDS11(x[13]) ^ x[19];
		x[14] = ROTATEUPWARDS11(x[14]) ^ x[16];
		x[15] = ROTATEUPWARDS11(x[15]) ^ x[17];

		/* "add x_0jklm into x_1~j~k~l~m modulo 2^32" */
		x[31] += x[0];
		x[30] += x[1];
		x[29] += x[2];
		x[28] += x[3];
		x[27] += x[4];
		x[26] += x[5];
		x[25] += x[6];
		x[24] += x[7];
		x[23] += x[8];
		x[22] += x[9];
		x[21] += x[10];
		x[20] += x[11];
		x[19] += x[12];
		x[18] += x[13];
		x[17] += x[14];
		x[16] += x[15];

		/* "rotate x_0jklm upwards by 7 bits" */
		/* "xor x_1j~k~l~m into x_0jklm" */
		x[0] = ROTATEUPWARDS7(x[0]) ^ x[23];
		x[1] = ROTATEUPWARDS7(x[1]) ^ x[22];
		x[2] = ROTATEUPWARDS7(x[2]) ^ x[21];
		x[3] = ROTATEUPWARDS7(x[3]) ^ x[20];
		x[4] = ROTATEUPWARDS7(x[4]) ^ x[19];
		x[5] = ROTATEUPWARDS7(x[5]) ^ x[18];
		x[6] = ROTATEUPWARDS7(x[6]) ^ x[17];
		x[7] = ROTATEUPWARDS7(x[7]) ^ x[16];
		x[8] = ROTATEUPWARDS7(x[8]) ^ x[31];
		x[9] = ROTATEUPWARDS7(x[9]) ^ x[30];
		x[10] = ROTATEUPWARDS7(x[10]) ^ x[29];
		x[11] = ROTATEUPWARDS7(x[11]) ^ x[28];
		x[12] = ROTATEUPWARDS7(x[12]) ^ x[27];
		x[13] = ROTATEUPWARDS7(x[13]) ^ x[26];
		x[14] = ROTATEUPWARDS7(x[14]) ^ x[25];
		x[15] = ROTATEUPWARDS7(x[15]) ^ x[24];

		/* "add x_0jklm into x_1j~kl~m modulo 2^32" */
		x[21] += x[0];
		x[20] += x[1];
		x[23] += x[2];
		x[22] += x[3];
		x[17] += x[4];
		x[16] += x[5];
		x[19] += x[6];
		x[18] += x[7];
		x[29] += x[8];
		x[28] += x[9];
		x[31] += x[10];
		x[30] += x[11];
		x[25] += x[12];
		x[24] += x[13];
		x[27] += x[14];
		x[26] += x[15];

		/* "rotate x_0jklm upwards by 11 bits" */
		/* "xor x_1jkl~m into x_0jklm" */
		x[0] = ROTATEUPWARDS11(x[0]) ^ x[17];
		x[1] = ROTATEUPWARDS11(x[1]) ^ x[16];
		x[2] = ROTATEUPWARDS11(x[2]) ^ x[19];
		x[3] = ROTATEUPWARDS11(x[3]) ^ x[18];
		x[4] = ROTATEUPWARDS11(x[4]) ^ x[21];
		x[5] = ROTATEUPWARDS11(x[5]) ^ x[20];
		x[6] = ROTATEUPWARDS11(x[6]) ^ x[23];
		x[7] = ROTATEUPWARDS11(x[7]) ^ x[22];
		x[8] = ROTATEUPWARDS11(x[8]) ^ x[25];
		x[9] = ROTATEUPWARDS11(x[9]) ^ x[24];
		x[10] = ROTATEUPWARDS11(x[10]) ^ x[27];
		x[11] = ROTATEUPWARDS11(x[11]) ^ x[26];
		x[12] = ROTATEUPWARDS11(x[12]) ^ x[29];
		x[13] = ROTATEUPWARDS11(x[13]) ^ x[28];
		x[14] = ROTATEUPWARDS11(x[14]) ^ x[31];
		x[15] = ROTATEUPWARDS11(x[15]) ^ x[30];
	}
}

__global__	__launch_bounds__(TPB, BPM)
void cubehash256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 Hash[4];

		Hash[0] = __ldg(&g_hash[0 * threads + thread]);
		Hash[1] = __ldg(&g_hash[1 * threads + thread]);
		Hash[2] = __ldg(&g_hash[2 * threads + thread]);
		Hash[3] = __ldg(&g_hash[3 * threads + thread]);

		uint32_t x[32] =
		{
			0xEA2BD4B4, 0xCCD6F29F, 0x63117E71, 0x35481EAE,
			0x22512D5B, 0xE5D94E63, 0x7E624131, 0xF4CC12BE,
			0xC2D0B696, 0x42AF2070, 0xD0720C35, 0x3361DA8C,
			0x28CCECA4, 0x8EF8AD83, 0x4680AC00, 0x40E5FBAB,
			0xD89041C3, 0x6107FBD5, 0x6C859D41, 0xF0B26679,
			0x09392549, 0x5FA25603, 0x65C892FD, 0x93CB6285,
			0x2AF2B5AE, 0x9E4B4E60, 0x774ABFDD, 0x85254725,
			0x15815AEB, 0x4AB6AAD6, 0x9CDAF8AF, 0xD6032C0A
		};

		x[0] ^= Hash[0].x;
		x[1] ^= Hash[0].y;
		x[2] ^= Hash[1].x;
		x[3] ^= Hash[1].y;
		x[4] ^= Hash[2].x;
		x[5] ^= Hash[2].y;
		x[6] ^= Hash[3].x;
		x[7] ^= Hash[3].y;

		rrounds(x);
		x[0] ^= 0x80U;
		rrounds(x);

		/* "the integer 1 is xored into the last state word x_11111" */
		x[31] ^= 1U;

		/* "the state is then transformed invertibly through 10r identical rounds" */
		for (int i = 0; i < 10; ++i) rrounds(x);

		g_hash[0 * threads + thread] = make_uint2(x[0], x[1]);
		g_hash[1 * threads + thread] = make_uint2(x[2], x[3]);
		g_hash[2 * threads + thread] = make_uint2(x[4], x[5]);
		g_hash[3 * threads + thread] = make_uint2(x[6], x[7]);

	}
}

__host__
void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb;

	if (cuda_arch[dev_id] >= 750) tpb = TPB60;
	else if (cuda_arch[dev_id] >= 610) tpb = TPB52;
	else if (cuda_arch[dev_id] >= 600) tpb = TPB60;
	else if (cuda_arch[dev_id] >= 520) tpb = TPB52;
	else if (cuda_arch[dev_id] >= 500) tpb = TPB50;
	else if (cuda_arch[dev_id] >= 300) tpb = TPB30;
	else tpb = TPB20;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	cubehash256_gpu_hash_32 << <grid, block, 0, gpustream[thr_id] >> > (threads, startNounce, (uint2*)d_hash);
	CUDA_SAFE_CALL(cudaGetLastError());
	if (opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
