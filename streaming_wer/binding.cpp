#include "streaming_wer.h"

#include <torch/types.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor StreamingLevenshteinDistance(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length,
        torch::Tensor source_delays,
        torch::Tensor target_delays,
        float threshold,
        int ins_cost, 
        int del_cost, 
        int sub_cost, 
        int str_cost) {

    CHECK_INPUT(source);
    CHECK_INPUT(target);
    CHECK_INPUT(source_length);
    CHECK_INPUT(target_length);
    CHECK_INPUT(source_delays);
    CHECK_INPUT(target_delays);

    return StreamingLevenshteinDistanceCuda(source, target, source_length, target_length, source_delays, target_delays, threshold, ins_cost, del_cost, sub_cost, str_cost);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("streaming_levenshtein_distance", &StreamingLevenshteinDistance, "Streaming Levenshtein distance");
}
