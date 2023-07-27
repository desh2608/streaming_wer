#include "streaming_wer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c10/cuda/CUDAStream.h>



template <typename scalar_t>
__global__ void streaming_levenshtein_distance_kernel(
        const int* __restrict__ source,
        const int* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const float* __restrict__ source_delays,
        const float* __restrict__ target_delays,
        const size_t source_size,
        const size_t target_size,
        const float threshold,
        const int ins_cost, const int del_cost, const int sub_cost, const int str_cost,
        int* __restrict__ operations) {

    extern __shared__ short err[];

    const int i = blockIdx.x;

    auto err_prev = err;
    auto err_curr = err + (target_size + 1);

    const int* hyp_begin = source + i * source_size;
    const int* ref_begin = target + i * target_size;
    const float* hyp_delays_begin = source_delays + i * source_size;
    const float* ref_delays_begin = target_delays + i * target_size;

    int hyp_size = source_length[i];
    int ref_size = target_length[i];

    for (int r = 0; r <= ref_size; ++r) {
        err_prev[r] = r * del_cost; // total_cost = del_cost
    }

    for (int h = 1; h <= hyp_size; ++h) {

        err_curr[0] = err_prev[0] + ins_cost;    // total_cost = ins_cost

        auto hyp = hyp_begin + h - 1;
        auto hyp_delay = hyp_delays_begin + h - 1;

        for (int r = 1; r <= ref_size; ++r) {

            int ins_err = err_prev[r] + ins_cost;
            int del_err = err_curr[r-1] + del_cost;
            int sub_err;

            auto ref = ref_begin + r - 1;
            auto ref_delay = ref_delays_begin + r - 1;

            if (*hyp == *ref) {
                if (*hyp_delay - *ref_delay <= threshold) {
                    sub_err = err_prev[r-1];
                } else {
                    sub_err = err_prev[r-1] + str_cost;
                }
            } else {
                sub_err = err_prev[r-1] + sub_cost;
            }

            if (sub_err < ins_err && sub_err < del_err) {

                err_curr[r] = sub_err;                  // total_cost

            } else if (del_err < ins_err) {

                err_curr[r] = del_err;                  // total_cost

            } else {

                err_curr[r] = ins_err;                  // total_cost
            }

        }

        // alternate for the next recursion
        short* temp = err_prev;
        err_prev = err_curr;
        err_curr = temp;
    }

    operations[i] = err_prev[ref_size];
}



torch::Tensor StreamingLevenshteinDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length,
        torch::Tensor source_delays,
        torch::Tensor target_delays,
        float threshold,
        int ins_cost, int del_cost, int sub_cost, int str_cost) {

    const auto batch_size = source.size(0);
    const auto shared_size = (target.size(1) + 1) * 2 * sizeof(short);

    at::TensorOptions options(source.device());

    options = options.dtype(at::ScalarType::Int);

    auto operations = torch::empty(batch_size, options);

    auto stream = c10::cuda::getCurrentCUDAStream(source.device().index());

    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "streaming_levenshtein_distance", ([&] {
        streaming_levenshtein_distance_kernel<int><<<batch_size, 1, shared_size, stream>>>(
            source.data<int>(),
            target.data<int>(),
            source_length.data<int>(),
            target_length.data<int>(),
            source_delays.data<float>(),
            target_delays.data<float>(),
            source.size(1),
            target.size(1),
            threshold,
            ins_cost, del_cost, sub_cost, str_cost,
            operations.data<int>());
    }));

    return operations;
}
