#include "miner.h"
#include "cuda_helper.h"
extern "C" {
#include "sph/yescrypt.h"
}

#include <math.h>
extern "C" {
#include "SHA3api_ref.h"
}

extern void yescrypt_cpu_init(int thr_id, int threads, uint32_t *d_hash1, uint32_t *d_hash2, uint32_t *d_hash3, uint32_t *d_hash4);
extern void yescrypt_setTarget(int thr_id, uint32_t pdata[20], char *key, uint32_t key_len, const int perslen); 
extern void yescrypt_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resultnonces, uint32_t target, const uint32_t N, const uint32_t r, const uint32_t p, const int is112); 
extern void yescrypt_cpu_free(int thr_id); 

extern char *yescrypt_key;
extern size_t yescrypt_key_len;
extern uint32_t yescrypt_param_N;
extern uint32_t yescrypt_param_r;
extern uint32_t yescrypt_param_p;

void yescrypt_hash_base(void *state, const void *input, const uint32_t N, const uint32_t r, const uint32_t p, char *key, const size_t key_len, int perslen) 
{
	if (client_key_len == 0xff)
	{
		client_key = key;
		client_key_len = key_len;
	}
	yescrypt_bsty((unsigned char*)input, perslen, (unsigned char*)input, perslen, N, r, p, (unsigned char *)state, 32);
}

void yescrypt_hash(void *state, const void *input, int perslen)
{
	yescrypt_hash_base(state, input, 2048, 8, 1, NULL, 0, perslen);
}

void yescryptr8_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 2048, 8, 1, (char *)"Client Key", 10, 80);
}

void yescryptr16_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 16, 1, (char *)"Client Key", 10, 80);
}

void yescryptr16v2_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 16, 4, (char *)"PPTPPubKey", 10, 80);
}

void yescryptr24_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 24, 1, (char *)"Jagaricoin", 10, 80);
}

void yescryptr32_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 32, 1, (char *)"WaviBanana", 10, 80);
}

int scanhash_yescrypt_base(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce, uint32_t *hashes_done,
	const uint32_t N, const uint32_t r, const uint32_t p,
	char *key, const size_t key_len, int perslen) {
	static THREAD uint32_t *d_hash1 = nullptr;
	static THREAD uint32_t *d_hash2 = nullptr;
	static THREAD uint32_t *d_hash3 = nullptr;
	static THREAD uint32_t *d_hash4 = nullptr;

	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, dev_id);

	uint32_t CUDAcore_count;
	if (device_sm[dev_id] == 600)		// Pascal(P100)
		CUDAcore_count = props.multiProcessorCount * 64;
	else if (device_sm[dev_id] >= 500)	// Maxwell/Pascal(other)/Volta
		CUDAcore_count = props.multiProcessorCount * 128;
	else if (device_sm[dev_id] >= 300)	// Kepler
		CUDAcore_count = props.multiProcessorCount * 96; // * 192
	else if (device_sm[dev_id] >= 210)	// Fermi(GF11x)
		CUDAcore_count = props.multiProcessorCount * 48;
	else					// Fermi(GF10x)
		CUDAcore_count = props.multiProcessorCount * 32;

	uint32_t throughputmax;
#if defined WIN32 && !defined _WIN64
	// 2GB limit for cudaMalloc
	uint32_t max_thread_multiple = (min(0x7fffffffULL, props.totalGlobalMem) - 256 * 1024 * 1024) / (((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)) * CUDAcore_count);
#else
	uint32_t max_thread_multiple = (props.totalGlobalMem - 256 * 1024 * 1024) / (((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)) * CUDAcore_count);
#endif

	if (device_sm[dev_id] > 500)		// Maxwell(GTX9xx)/Pascal/Volta
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * min(3, max_thread_multiple));
	else if (device_sm[dev_id] == 500)	// Maxwell(GTX750Ti/GTX750)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * min(2, max_thread_multiple));
	else if (device_sm[dev_id] >= 300)	// Kepler
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count);
	else if (device_sm[dev_id] >= 210)	// Fermi(GF11x)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * min(2, max_thread_multiple));
	else								// Fermi(GF10x)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * min(2, max_thread_multiple));

	throughputmax = (throughputmax / CUDAcore_count) * CUDAcore_count;
	if (throughputmax == 0) throughputmax = CUDAcore_count;

	uint32_t throughput = min(throughputmax, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	static THREAD bool init = false;
	if (!init)
	{
		applog(LOG_WARNING, "Using intensity %.3f (%d threads)", throughput2intensity(throughputmax), throughputmax);
		CUDA_SAFE_CALL(cudaSetDevice(dev_id));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
#if defined WIN32 && !defined _WIN64
		// 2GB limit for cudaMalloc
		if (throughputmax > (min(0x7fffffffULL, props.totalGlobalMem) - 256 * 1024 * 1024) / ((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)))
#else
		if (throughputmax > (props.totalGlobalMem - 256 * 1024 * 1024) / ((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)))
#endif
		{
			applog(LOG_ERR, "Memory Error");
			mining_has_stopped[thr_id] = true;
			cudaStreamDestroy(gpustream[thr_id]);
			proper_exit(2);
		}

		size_t hash1_sz = 2 * 16 * r * p * sizeof(uint32_t);	// B
		size_t hash2_sz = 512 * sizeof(uint32_t);				// S(4way)
		size_t hash3_sz = 2 * N * r * sizeof(uint32_t);			// V(16way)
		size_t hash4_sz = 8 * sizeof(uint32_t);					// sha256
		CUDA_SAFE_CALL(cudaMalloc(&d_hash1, hash1_sz * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2, hash2_sz * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash3, hash3_sz * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash4, hash4_sz * throughputmax));

		yescrypt_cpu_init(thr_id, throughputmax, d_hash1, d_hash2, d_hash3, d_hash4);
		mining_has_stopped[thr_id] = false;

		init = true;
	}

	uint32_t endiandata[32]; 
	for (int k = 0; k < 32; k++) 
		be32enc(&endiandata[k], pdata[k]);

	yescrypt_setTarget(thr_id, pdata, key, key_len, perslen);

	do {
		uint32_t foundNonce[2] = { 0, 0 };

		yescrypt_cpu_hash_32(thr_id, throughput, pdata[19], foundNonce, ptarget[7], N, r, p, perslen == 112 ? 1 : 0);

		if (stop_mining)
		{
			mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (foundNonce[0] != 0 && foundNonce[0] < max_nonce)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash64[8] = { 0 };
			if (opt_verify)
			{
				be32enc(&endiandata[19], foundNonce[0]);
				yescrypt_hash_base(vhash64, endiandata, N, r, p, key, key_len, perslen);
			}
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (foundNonce[1] != 0 && foundNonce[1] < max_nonce) {
					if (opt_verify)
					{
						be32enc(&endiandata[19], foundNonce[1]);
						yescrypt_hash_base(vhash64, endiandata, N, r, p, key, key_len, perslen);
					}
					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						if (perslen == 80) 
							pdata[21] = foundNonce[1]; 
						else 
							pdata[29] = foundNonce[1]; 
						res++; 
						if (opt_benchmark)  applog(LOG_INFO, "GPU #%d Found second nonce %08x", thr_id, foundNonce[1]);
					}
					else
					{
						if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
							applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", dev_id);
					}
				}
				pdata[19] = foundNonce[0];
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d Found nonce % 08x", thr_id, foundNonce[0]);
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", dev_id);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

int scanhash_yescrypt(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done, int perslen)
{
	if (yescrypt_param_N == 0) yescrypt_param_N = 2048;
	if (yescrypt_param_r == 0) yescrypt_param_r = 8;
	if (yescrypt_param_p == 0) yescrypt_param_p = 1;
	return scanhash_yescrypt_base(thr_id, pdata, ptarget, max_nonce, hashes_done, yescrypt_param_N, yescrypt_param_r, yescrypt_param_p, yescrypt_key, yescrypt_key_len, perslen); 
} 

int scanhash_yescryptr8(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	return scanhash_yescrypt_base(thr_id, pdata, ptarget, max_nonce, hashes_done, 2048, 8, 1, (char *)"Client Key", 10, 80);
}

int scanhash_yescryptr16(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	return scanhash_yescrypt_base(thr_id, pdata, ptarget, max_nonce, hashes_done, 4096, 16, 1, (char *)"Client Key", 10, 80);
}

int scanhash_yescryptr16v2(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, pdata, ptarget, max_nonce, hashes_done, 4096, 16, 4, (char *)"PPTPPubKey", 10, 80);
}

int scanhash_yescryptr24(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, pdata, ptarget, max_nonce, hashes_done, 4096, 24, 1, (char *)"Jagaricoin", 10, 80);
}

int scanhash_yescryptr32(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, pdata, ptarget, max_nonce, hashes_done, 4096, 32, 1, (char *)"WaviBanana", 10, 80);
}
