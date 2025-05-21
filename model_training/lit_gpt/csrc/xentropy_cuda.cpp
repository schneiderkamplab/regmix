#include <torch/extension.h>

// Declaration of CUDA functions
std::vector<torch::Tensor> xentropy_forward_cuda(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    double smoothing);

torch::Tensor xentropy_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& logits,
    const torch::Tensor& lse,
    const torch::Tensor& labels,
    double smoothing);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &xentropy_forward_cuda, "Fused CrossEntropy forward (CUDA)");
    m.def("backward", &xentropy_backward_cuda, "Fused CrossEntropy backward (CUDA)");
}
