// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <ff/pasta.hpp>

typedef jacobian_t<pallas_t> point_t;
typedef xyzz_t<pallas_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef vesta_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__
extern "C"
RustError cuda_pippenger_pallas(point_t *out, const affine_t points[], size_t npoints,
                                              const scalar_t scalars[])
{   return mult_pippenger<bucket_t>(out, points, npoints, scalars);   }

using pallas_msm_t = msm_t<bucket_t, point_t, affine_t, scalar_t>;

struct pallas_msm_context {
    int device;
    size_t max_points;
    pallas_msm_t msm;

    static int current_device()
    {
        int device;
        CUDA_OK(cudaGetDevice(&device));
        return device;
    }

    pallas_msm_context(const affine_t points[], size_t npoints)
        : device(current_device()), max_points(npoints), msm(points, npoints)
    {
        CUDA_OK(cudaDeviceSynchronize());
    }
};

extern "C"
RustError cuda_pippenger_pallas_context_create(
    void** context,
    const affine_t points[],
    size_t npoints)
{
    if (context == nullptr || points == nullptr || npoints == 0) {
        return RustError{cudaErrorInvalidValue};
    }

    *context = nullptr;

    try {
        *context = new pallas_msm_context(points, npoints);
        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

extern "C"
RustError cuda_pippenger_pallas_context_invoke(
    void *context,
    point_t *out,
    const scalar_t scalars[],
    size_t npoints)
{
    if (context == nullptr || out == nullptr || scalars == nullptr ||
        npoints == 0) {
        return RustError{cudaErrorInvalidValue};
    }

    auto *prepared_ctx = static_cast<pallas_msm_context *>(context);

    const auto device_status = cudaSetDevice(prepared_ctx->device);
    if (device_status != cudaSuccess) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{device_status, cudaGetErrorString(device_status)};
#else
        return RustError{device_status};
#endif
    }

    if (npoints > prepared_ctx->max_points) {
        return RustError{cudaErrorInvalidValue};
    }

    // Passing nullptr in the points field tells sppark to use the bases
    // uploaded by the context constructor
    return prepared_ctx->msm.invoke(
        *out, static_cast<const affine_t *>(nullptr), npoints, scalars, true);
}

extern "C"
void cuda_pippenger_pallas_context_destroy(void *context)
{
    if (context == nullptr) {
        return;
    }

    auto *prepared_ctx = static_cast<pallas_msm_context *>(context);
    if (cudaSetDevice(prepared_ctx->device) != cudaSuccess) {
        return;
    }
    delete prepared_ctx;
}

#endif
