#pragma once

#include <torch/extension.h>

torch::Tensor StreamingLevenshteinDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length,
        torch::Tensor source_delays,
        torch::Tensor target_delays,
        float threshold,
        int ins_cost, int del_cost, int sub_cost, int str_cost);
