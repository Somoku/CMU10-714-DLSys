#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t iterations = (m + batch - 1) / batch;
    for (size_t iter = 0; iter < iterations; ++iter) {
        size_t batch_loc = iter * batch;
        size_t batch_size = fmin(batch, m - batch_loc);
        // Z_batch computation
        float *Z = new float[batch_size * k];
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float tmp = 0;
                for (size_t t = 0; t < n; ++t) 
                    tmp += X[batch_loc * n + i * n + t] * theta[t * k + j];
                Z[i * k + j] = exp(tmp);
            }
        }
        for (size_t i = 0; i < batch_size; ++i) {
            float line_sum = 0;
            for (size_t j = 0; j < k; ++j)
                line_sum += Z[i * k + j];
            for (size_t j = 0; j < k; ++j)
                Z[i * k + j] /= line_sum;
        }
        // Z_batch - I_batch compuation
        for (size_t i = 0; i < batch_size; ++i) 
            Z[i * k + y[batch_loc + i]] -= 1;
        // Gradient computation
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float tmp = 0;
                for (size_t t = 0; t < batch_size; t++) 
                    tmp += X[t * n + i + batch_loc * n] * Z[t * k + j];
                theta[i * k + j] -= lr / batch * tmp;
            }
        }
        delete[] Z;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
