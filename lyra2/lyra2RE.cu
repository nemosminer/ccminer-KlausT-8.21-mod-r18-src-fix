extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"
extern "C" {
#include "SHA3api_ref.h"
}
extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void blake256_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void keccak256_cpu_init(int thr_id, uint32_t threads);
extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void skein256_cpu_init(int thr_id, uint32_t threads);
extern void groestl256_setTarget(int thr_id, const void *ptarget);
extern void groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t *resultnonces);
extern void groestl256_cpu_init(int thr_id, uint32_t threads);

extern void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint64_t *Hash);

extern void lyra2_cpu_init(int thr_id, uint32_t threads, const uint32_t nRows, const uint32_t nCols, uint64_t *d_state);

extern void lyra2_cpu_free(int thr_id);

void lyra2re_hash(void *state, const void *input)
{
	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_skein256_context     ctx_skein;
	sph_groestl256_context   ctx_groestl;

	uint32_t hashA[8], hashB[8];

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8, LYRA2_BUG,0);
	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}

int scanhash_lyra2re(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	static THREAD uint64_t *d_hash = nullptr;
	static THREAD uint64_t *d_hash2 = nullptr;

	const uint32_t first_nonce = pdata[19];
	
	int dev_id = device_map[thr_id];

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);

	uint32_t CUDAcore_count;
	if (device_sm[dev_id] == 600)		// Pascal(P100)
		CUDAcore_count = props.multiProcessorCount * 64;
	else if (device_sm[dev_id] >= 500)	// Maxwell/Pascal(other)/Volta
		CUDAcore_count = props.multiProcessorCount * 128;
	else if (device_sm[dev_id] >= 300)	// Kepler
		CUDAcore_count = props.multiProcessorCount * 192;
	else if (device_sm[dev_id] >= 210)	// Fermi(GF11x)
		CUDAcore_count = props.multiProcessorCount * 48;
	else					// Fermi(GF10x)
		CUDAcore_count = props.multiProcessorCount * 32;

	uint32_t throughputmax;

	if (device_sm[dev_id] > 500)		// Maxwell(GTX9xx)/Pascal/Volta
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * 339);
	else if (device_sm[dev_id] == 500)	// Maxwell(GTX750Ti/GTX750)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * 226);
	else if (device_sm[dev_id] >= 300)	// Kepler
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * 87);
	else if (device_sm[dev_id] >= 210)	// Fermi(GF11x)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count);
	else								// Fermi(GF10x)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count);

	throughputmax = (throughputmax / CUDAcore_count) * CUDAcore_count;
	if (throughputmax == 0) throughputmax = CUDAcore_count;

	uint32_t throughput = min(throughputmax, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x004f;

	static THREAD bool init = false;
	if (!init)
	{ 
		applog(LOG_WARNING, "Using intensity %2.2f (%d threads)", throughput2intensity(throughputmax), throughputmax);
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));

		CUDA_SAFE_CALL(cudaMalloc(&d_hash2, 4ULL * 4 * sizeof(uint64_t) * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash, 8ULL * sizeof(uint32_t) * throughputmax));

		keccak256_cpu_init(thr_id, throughput);
		skein256_cpu_init(thr_id, throughput);
		groestl256_cpu_init(thr_id, throughput);
		lyra2_cpu_init(thr_id, throughput, 8, 8, d_hash2);
		mining_has_stopped[thr_id] = false;

		init = true; 
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	blake256_cpu_setBlock_80(thr_id, pdata);
	groestl256_setTarget(thr_id, ptarget);

	do {
		uint32_t foundNonce[2] = { 0, 0 };

		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash);
		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash);
		lyra2_cpu_hash_32(thr_id, throughput, pdata[19], 1, 8, 8, LYRA2_BUG, d_hash);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash);
		groestl256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash, foundNonce);

		if(stop_mining)
		{
			mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);
		}
		if(foundNonce[0] != 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8]={0};
			if(opt_verify)
			{
				be32enc(&endiandata[19], foundNonce[0]);
				lyra2re_hash(vhash64, endiandata);
			}
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (foundNonce[1] != 0)
				{
					if(opt_verify)
					{
						be32enc(&endiandata[19], foundNonce[1]);
						lyra2re_hash(vhash64, endiandata);
					}
					if(vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundNonce[1];
						res++;
						if(opt_benchmark)  applog(LOG_INFO, "GPU #%d Found second nonce %08x", thr_id, foundNonce[1]);
					}
					else
					{
						if(vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
							applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", device_map[thr_id]);
					}
				}
				pdata[19] = foundNonce[0];
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d Found nonce % 08x", thr_id, foundNonce[0]);
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", device_map[thr_id]);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce ;
	return 0;
}
