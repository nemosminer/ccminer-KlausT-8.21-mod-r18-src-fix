extern "C" {
#include "lyra2/Lyra2.h"
}

#include <miner.h>
#include <cuda_helper.h>

static uint64_t* d_state[MAX_GPUS];

extern void lyra2_cpu_hash_80_ending(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, const uint32_t bug, uint32_t *resultnonces);

extern void lyra2_cpu_init(int thr_id, uint32_t threads, const uint32_t nRows, const uint32_t nCols, uint64_t *d_state);

extern void lyra2_setData(const void *data);
extern void lyra2_cpu_free(int thr_id);

void lyra2z330_hash(void *state, const void *input, const uint64_t timeCost, const uint64_t row, const uint64_t col, const uint32_t bug)
{
	uint32_t hashA[8];

	LYRA2(hashA, 32, input, 80, input, 80, timeCost, row, col, bug,0);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_lyra2_base(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce, uint32_t *hashes_done,
	const uint64_t timeCost, const uint64_t row, const uint64_t col, const uint32_t bug) {
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	static THREAD uint64_t *d_hash = nullptr;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, dev_id);

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
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * 3);
	else if (device_sm[dev_id] == 500)	// Maxwell(GTX750Ti/GTX750)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * 3);
	else if (device_sm[dev_id] >= 300)	// Kepler
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count);
	else if (device_sm[dev_id] >= 210)	// Fermi(GF11x)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count);
	else								// Fermi(GF10x)
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count);

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

		size_t state_sz = sizeof(uint64_t) * 16;

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_state[thr_id], state_sz * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash, 8ULL * sizeof(uint32_t) * throughputmax));
		lyra2_cpu_init(thr_id, throughputmax, 330, 256, d_state[thr_id]);

		api_set_throughput(thr_id, throughput);
		init = true;
	}


	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	lyra2_setData(endiandata);

	do {
		uint32_t foundNonce[2] = { 0, 0 };
		lyra2_cpu_hash_80_ending(thr_id, throughput, pdata[19], ptarget[7], timeCost, row, col, bug, foundNonce);

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
				lyra2z330_hash(vhash64, endiandata, timeCost, row, col, bug);
			}
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (foundNonce[1] != 0 && foundNonce[1] < max_nonce) {
					if (opt_verify)
					{
						be32enc(&endiandata[19], foundNonce[1]);
						lyra2z330_hash(vhash64, endiandata, timeCost, row, col, bug);
					}
					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundNonce[1];
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

int scanhash_lyra2z330(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	return scanhash_lyra2_base(thr_id, pdata, ptarget, max_nonce, hashes_done, 2, 330, 256, LYRA2_NOBUG);
}

// cleanup
extern "C" void free_lyra2z330(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_state[thr_id]);

	lyra2_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
