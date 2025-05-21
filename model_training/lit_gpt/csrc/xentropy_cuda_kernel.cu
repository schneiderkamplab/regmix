#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void xentropy_forward_kernel(
    const scalar_t* logits,
    const int64_t* labels,
    scalar_t* losses,
    scalar_t* lse,
    int batch_size,
    int vocab_size,
    double smoothing
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const scalar_t* logits_row = logits + idx * vocab_size;
    
    // Find max for numerical stability
    scalar_t max_logit = logits_row[0];
    for (int i = 1; i < vocab_size; i++) {
        max_logit = max(max_logit, logits_row[i]);
    }

    // Compute log sum exp
    scalar_t sum_exp = 0;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += exp(logits_row[i] - max_logit);
    }
    lse[idx] = log(sum_exp) + max_logit;

    // Compute loss
    int label = labels[idx];
    if (label < 0 || label >= vocab_size) {
        losses[idx] = 0;
        return;
    }

    if (smoothing == 0) {
        losses[idx] = -logits_row[label] + lse[idx];
    } else {
        losses[idx] = (1 - smoothing) * (-logits_row[label] + lse[idx]) - 
                     smoothing * lse[idx];
    }
}

std::vector<torch::Tensor> xentropy_forward_cuda(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    double smoothing
) {
    const auto batch_size = logits.size(0);
    const auto vocab_size = logits.size(1);

    auto losses = torch::empty({batch_size}, logits.options());
    auto lse = torch::empty({batch_size}, logits.options());

    const int threads = 1024;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "xentropy_forward_cuda", ([&] {
        xentropy_forward_kernel<scalar_t><<<blocks, threads>>>(
            logits.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            losses.data_ptr<scalar_t>(),
            lse.data_ptr<scalar_t>(),
            batch_size,
            vocab_size,
            smoothing
        );
    }));

    return {losses, lse};
}

template <typename scalar_t>
__global__ void xentropy_backward_kernel(
    const scalar_t* grad_output,
    const scalar_t* logits,
    const scalar_t* lse,
    const int64_t* labels,
    scalar_t* grad_logits,
    int batch_size,
    int vocab_size,
    double smoothing
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int label = labels[idx];
    if (label < 0 || label >= vocab_size) return;

    const scalar_t* logits_row = logits + idx * vocab_size;
    scalar_t* grad_logits_row = grad_logits + idx * vocab_size;
    const scalar_t grad = grad_output[idx];

    for (int i = 0; i < vocab_size; i++) {
        const scalar_t exp_logit = exp(logits_row[i] - lse[idx]);
        if (i == label) {
            grad_logits_row[i] = grad * ((smoothing - 1) + smoothing * exp_logit);
        } else {
            grad_logits_row[i] = grad * smoothing * exp_logit;
        }
    }
}

torch::Tensor xentropy_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& logits,
    const torch::Tensor& lse,
    const torch::Tensor& labels,
    double smoothing
) {
    const auto batch_size = logits.size(0);
    const auto vocab_size = logits.size(1);

    auto grad_logits = torch::zeros_like(logits);

    const int threads = 1024;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "xentropy_backward_cuda", ([&] {
        xentropy_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            logits.data_ptr<scalar_t>(),
            lse.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            grad_logits.data_ptr<scalar_t>(),
            batch_size,
            vocab_size,
            smoothing
        );
    }));

    return grad_logits;
}
