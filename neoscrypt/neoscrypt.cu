#include <string.h>
#include "cuda_helper.h"
#include "miner.h"
#include "sph/neoscrypt.h"

//extern void neoscrypt_cpu_hash_k4_52(int stratum, int thr_id, int threads, uint32_t startNounce, int order, uint32_t* foundnonce);
extern void neoscrypt_init(int thr_id, uint32_t threads);
extern void neoscrypt_setBlockTarget(uint32_t thr_id, uint32_t* const pdata, uint32_t* const target);
extern void neoscrypt_hash_k4(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces, bool stratum, uint32_t bpm);

int scanhash_neoscrypt(bool stratum, int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	static THREAD uint32_t hw_errors = 0;
	static THREAD uint32_t *foundNonce = nullptr;
	
	if (opt_benchmark)
	{
		ptarget[7] = 0x01ff;
		stratum = 0;
	}

	cudaDeviceProp props;
	int dev_id = device_map[thr_id % MAX_GPUS];
	cudaGetDeviceProperties(&props, dev_id);
	double intensity = 0.001 * (double)props.multiProcessorCount * (double)_ConvertSMVer2Cores(props.major, props.minor) *(double)props.clockRate;

	uint32_t bpm = (uint32_t)((double)props.l2CacheSize / ((double)(props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor) * 256) * 1.2));

	// 1way-mode
	// GTX1080Ti       : 1765kH/s (3584 Core, 1683MHz) :  1,765,000[H/s] * 0.1[s] / 3584 / 1683 = 0.0293
	// GTX1060(6GB)    :  860kH/s (1280 Core, 1847MHz) :    860,000[H/s] * 0.1[s] / 1280 / 1847 = 0.0364
	// GTX980          :  558kH/s (2048 Core, 1304MHz) :    558,000[H/s] * 0.1[s] / 2048 / 1304 = 0.0209
	// GTX960          :  277kH/s (1024 Core, 1178MHz) :    277,000[H/s] * 0.1[s] / 1024 / 1178 = 0.0230
	// GTX750          :  144kH/s ( 512 Core, 1137MHz) :    144,000[H/s] * 0.1[s] /  512 / 1137 = 0.0247
	// 4way-mode
	// GT710           : 24.8kH/s ( 192 Core,  954MHz) :     24,900[H/s] * 0.1[s] /  192 /  954 = 0.0136
	// GT610           : 10.3kH/s (  48 Core, 1620MHz) :     10,300[H/s] * 0.1[s] /   48 / 1620 = 0.0132
	if (device_sm[dev_id] >= 600) intensity *= 0.0350;
	else if (device_sm[dev_id] >= 500) intensity *= 0.0220;
	else intensity *= 0.0136;

#if defined WIN32 && !defined _WIN64
	intensity = min(intensity, 57344);
#endif

	intensity = (double)((uint32_t)(throughput2intensity((uint32_t)intensity) * 4.0)) * 0.25;
	uint32_t throughputmax;
	throughputmax = (uint32_t)((1.0 + (intensity - (double)((uint32_t)intensity)))*(1UL << (int)intensity));
	throughputmax = device_intensity(dev_id, __func__, throughputmax);

	uint32_t throughput = min(throughputmax, max_nonce - first_nonce) & 0xffffff00;
	
	static THREAD bool init = false;

	if (!init)
	{
		intensity = throughput2intensity(throughputmax);
		applog(LOG_WARNING, "Using intensity %2.2f (%d threads)", intensity, throughputmax);

		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));

#if defined WIN32 && !defined _WIN64
		// 2GB limit for cudaMalloc
		if (throughputmax > 0x7fffffffULL / (32 * 128 * sizeof(uint64_t)))
		{
			applog(LOG_ERR, "intensity too high");
			mining_has_stopped[thr_id] = true;
			cudaStreamDestroy(gpustream[thr_id]);
			proper_exit(EXIT_FAILURE);
		}
#endif

		CUDA_SAFE_CALL(cudaMallocHost(&foundNonce, 2 * 4));

		neoscrypt_init(thr_id, throughputmax);

		mining_has_stopped[thr_id] = false;
		init = true;
	}

	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
	{
		if (stratum)
			be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
		else endiandata[k] = pdata[k];
	}

	neoscrypt_setBlockTarget(thr_id, endiandata, ptarget);

	do
	{
		neoscrypt_hash_k4(thr_id, throughput, pdata[19], foundNonce, stratum, bpm);

		if (stop_mining)
		{
			mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);
		}
		if (foundNonce[0] != 0xffffffff)
		{
			uint32_t vhash64[8] = { 0 };
			if (opt_verify)
			{
				if (stratum)
					be32enc(&endiandata[19], foundNonce[0]);
				else
					endiandata[19] = foundNonce[0];
				neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);
			}
			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
			{
				*hashes_done = pdata[19] - first_nonce + throughput;
				int res = 1;
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nonce %08x", device_map[thr_id], foundNonce[0]);
				pdata[19] = foundNonce[0];
				if (foundNonce[1] != 0xffffffff)
				{
					if (opt_verify)
					{
						if (stratum)
						{
							be32enc(&endiandata[19], foundNonce[1]);
						}
						else
						{
							endiandata[19] = foundNonce[1];
						}
						neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);
					}
					if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundNonce[1];
						res++;
						if (opt_benchmark)
							applog(LOG_INFO, "GPU #%d: Found second nonce %08x", device_map[thr_id], foundNonce[1]);
					}
					else
					{
						if (vhash64[7] != ptarget[7])
						{
							applog(LOG_WARNING, "GPU #%d: Second nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce[1]);
							hw_errors++;
						}
					}

				}
				return res;
			}
			else
			{
				if (vhash64[7] != ptarget[7])
				{
					applog(LOG_WARNING, "GPU #%d: Nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce[0]);
					hw_errors++;
				}
			}
			//						if(hw_errors > 0) applog(LOG_WARNING, "Hardware errors: %u", hw_errors);
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

