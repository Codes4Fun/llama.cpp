#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cl-common.h"

#define USE_INTERLEAVE

extern "C" {

typedef uint16_t ggml_fp16_t;

#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    int8_t  qs[QK8_0];     // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

}

static cl_program program;

static int ocl_mul_mat_init() {
    if (program) return 0;

    const char* source = NULL;
    FILE* file = fopen("cl-mul-mat.cl", "rb");
    if (file) {
        printf("source: loading...\n");
        fseek(file, 0, SEEK_END);
        size_t len = ftell(file);
        if (len > 20) {
            fseek(file, 0, SEEK_SET);
            char* buffer = new char[len + 1];
            if (fread(buffer, len, 1, file)) {
                printf("source: loaded\n");
                buffer[len] = 0;
                source = buffer;
            } else {
                printf("source: failed to read.\n");
                exit(0);
            }
        } else {
            printf("source: skipping, seems too small.\n");
            exit(0);
        }
        fclose(file);
    } else {
        printf("source: unabled to find cl-mul-mat.cl\n");
        exit(0);
    }

    // Create the compute program from the source buffer
    cl_int err;
    program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program! %s\n", getErrorString(err));
        exit(0);
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[16*1024];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        if (len > sizeof(buffer)) {
            char* buffer2 = new char[len+1];
            clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, len+1, buffer2, &len);
            printf("%s\n", buffer2);
        } else {
            printf("%s\n", buffer);
        }
        exit(0);
    }

    return 0;
}

static cl_kernel kernel_mul_mat_q4_q8;
static size_t kernel_mul_mat_q4_q8_local;

static cl_kernel kernel_mul_mat_q4_q8_half_raw;

static cl_kernel kernel_mul_mat_q4_q8_raw;

extern "C" int ocl_mul_mat_q4_q8_init() {
    if (kernel_mul_mat_q4_q8)
        return 0;

    if (ocl_mul_mat_init())
        return -1;

    cl_int err;

    // Create the compute kernel in the program we wish to run
    kernel_mul_mat_q4_q8 = clCreateKernel(program, "kernel_mul_mat_q4_q8", &err);
    if (!kernel_mul_mat_q4_q8 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! %s\n", getErrorString(err));
        exit(0);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel_mul_mat_q4_q8, device_ids[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_mul_mat_q4_q8_local), &kernel_mul_mat_q4_q8_local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(0);
    }

    // Create the compute kernel in the program we wish to run
    kernel_mul_mat_q4_q8_half_raw = clCreateKernel(program, "kernel_mul_mat_q4_q8_half_raw", &err);
    if (!kernel_mul_mat_q4_q8_half_raw || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! %s\n", getErrorString(err));
        exit(0);
    }

#ifdef USE_INTERLEAVE
    kernel_mul_mat_q4_q8_raw = clCreateKernel(program, "kernel_mul_mat_q4_q8_interleave", &err);
#else
    kernel_mul_mat_q4_q8_raw = clCreateKernel(program, "kernel_mul_mat_q4_q8_raw8", &err);
#endif
    if (!kernel_mul_mat_q4_q8_raw || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! %s\n", getErrorString(err));
        exit(0);
    }

    return 0;
}

extern "C" float table_f32_f16[1 << 16];

float vec32_dot_q4_q8(
    block_q4_0* x,
    block_q8_0* y,
    const unsigned int nb)
{
    float fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        int sum = 0;
        uint8_t* xqs = x->qs;
        int8_t* yqs0 = y->qs;
        int8_t* yqs1 = y->qs + (QK8_0/2);
        for (int i = 0; i < 16; i++) {
            uint8_t q4 = *(xqs++);
            int x0 = (q4 & 0xf) - 8;
            int x1 = (q4 >> 4) - 8;
            int y0 = *(yqs0++);
            int y1 = *(yqs1++);
            sum += x0 * y0 + x1 * y1;
        }
        fsum += sum * table_f32_f16[x->d] * table_f32_f16[y->d];
        x++; y++;
    }
    return fsum;
}

int mul_mat_q4_q8_validate(
    block_q4_0* vx,
    block_q8_0* vy,
    float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    int worst_k = 0;
    float worst_diff = 0;
    float worst_fsum = 0;
    for (unsigned int k = 0; k < dst_count; k++) {
        const div_t dyzw_x = div((int)k, dst_x_width);
        const div_t dzw_y = div(dyzw_x.quot, dst_y_width);
        const int x = dyzw_x.rem;
        const int y = dzw_y.rem;
        const int zw = dzw_y.quot;

        const int src0_index = x + src0_y_width * zw;
        block_q4_0* src0_data = (block_q4_0*)((char*)vx + (src0_index * src0_x_stride));

        const int src1_index = y + src1_y_width * zw;
        block_q8_0* src1_data = (block_q8_0*)((char*)vy + (src1_index * src1_x_stride));

        float fsum = vec32_dot_q4_q8(src0_data, src1_data, block_count);
        float diff = output[k] - fsum;
        if (fsum != 0.f) diff /= fsum;
        //if (fsum != output[k])
        if (diff > worst_diff) {
            worst_diff = diff;
            worst_k = k;
            worst_fsum = fsum;
        }
    }
    if (worst_diff > 0.34f)
    {
        printf("mismatch error! %f of %f %f\n", worst_diff, worst_fsum, output[worst_k]);
        static int logit = 0;
        if (logit) {
            const div_t dyzw_x = div((int)worst_k, dst_x_width);
            const div_t dzw_y = div(dyzw_x.quot, dst_y_width);
            const int x = dyzw_x.rem;
            const int y = dzw_y.rem;
            const int zw = dzw_y.quot;
            const int src0_index = x + src0_y_width * zw;
            block_q4_0* src0_data = (block_q4_0*)((char*)vx + (src0_index * src0_x_stride));
            const int src1_index = y + src1_y_width * zw;
            block_q8_0* src1_data = (block_q8_0*)((char*)vy + (src1_index * src1_x_stride));
            printf("test_q4_0 test_src0_data[] = {\n");
            for (unsigned int i = 0; i < dst_x_width / 32; i++) {
                printf("  {0x%x, {", *(uint16_t*)&src0_data[i].d);
                for (int j = 0; j < 16; j++) {
                    printf("%d,", src0_data[i].qs[j]);
                }
                printf("}},\n");
            }
            printf("};\ntest_q8_0 src1_data[] = \n{\n");
            for (unsigned int i = 0; i < dst_x_width / 32; i++) {
                printf("  {0x%x, {", *(uint16_t*)&src1_data[i].d);
                for (int j = 0; j < 32; j++) {
                    printf("%d,", src1_data[i].qs[j]);
                }
                printf("}},\n");
            }
            printf("};\n");
        }
        return 1;
    }
    return 0;
}

void mul_mat_q4_q8(
    block_q4_0* vx,
    block_q8_0* vy,
    float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    for (unsigned int k = 0; k < dst_count; k++) {
        const div_t dyzw_x = div((int)k, dst_x_width);
        const div_t dzw_y = div(dyzw_x.quot, dst_y_width);
        const int x = dyzw_x.rem;
        const int y = dzw_y.rem;
        const int zw = dzw_y.quot;

        const int src0_index = x + src0_y_width * zw;
        block_q4_0* src0_data = (block_q4_0*)((char*)vx + (src0_index * src0_x_stride));

        const int src1_index = y + src1_y_width * zw;
        block_q8_0* src1_data = (block_q8_0*)((char*)vy + (src1_index * src1_x_stride));

        output[k] = vec32_dot_q4_q8(src0_data, src1_data, block_count);
    }
}

#include <map>

typedef struct {
    cl_mem vx;
    cl_mem vxd; // for raw
} vx_cache_t;

static cl_mem mem_vx_scratch;
static cl_mem mem_vxd_scratch;
static std::map<block_q4_0*, vx_cache_t> mem_vx_cache;
static cl_mem mem_vy;
static cl_mem mem_vyd;
static cl_mem mem_output;
static unsigned int mem_vx_bytes = 0;
static unsigned int mem_vxd_bytes = 0;
static unsigned int mem_vy_bytes = 0;
static unsigned int mem_vyd_bytes = 0;
static unsigned int mem_output_bytes = 0;
static int src0_cache = 1;
static uint64_t mem_vx_cache_size = 0;

extern "C" void ocl_mul_mat_q4_q8(
    block_q4_0 * vx,
    block_q8_0 * vy,
    float* output,
    const unsigned int src0_bytes,
    const unsigned int src1_bytes,
    const unsigned int dst_bytes,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    /*int skip = 1;
    if (skip) {
        mul_mat_q4_q8(
            vx, vy, output,
            src0_y_width,
            src0_x_stride,
            src1_y_width,
            src1_x_stride,
            block_count,
            dst_x_width,
            dst_y_width,
            dst_count);
        return;
    }*/
    cl_int err;
    // allocate/reallocate scratch buffers if necessary
    cl_mem mem_vx;
    if (src0_cache) {
        if (!mem_vx_cache.count(vx)) {
            mem_vx = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_bytes, NULL, &err);
            err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_bytes, vx, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to write to source array!\n");
                exit(0);
            }
            mem_vx_cache[vx] = { mem_vx, NULL };
            mem_vx_cache_size += src0_bytes;
        }
        else {
            mem_vx = mem_vx_cache[vx].vx;
        }
    }
    else {
        if (mem_vx_bytes < src0_bytes) {
            if (mem_vx_bytes) {
                clReleaseMemObject(mem_vx_scratch);
            }
            mem_vx_scratch = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_bytes, NULL, &err);
            if (!mem_vx_scratch) {
                printf("Error: Failed to allocate device memory!\n");
                exit(0);
            }
            mem_vx_bytes = src0_bytes;
        }
        mem_vx = mem_vx_scratch;
        err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_bytes, vx, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(0);
        }
    }
    if (mem_vy_bytes < src1_bytes) {
        if (mem_vy_bytes) {
            clReleaseMemObject(mem_vy);
        }
        mem_vy = clCreateBuffer(context, CL_MEM_READ_ONLY, src1_bytes, NULL, &err);
        if (!mem_vy) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_vy_bytes = src1_bytes;
    }
    if (mem_output_bytes < dst_bytes) {
        if (mem_output_bytes) {
            clReleaseMemObject(mem_output);
        }
        mem_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_bytes, NULL, &err);
        if (!mem_output) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_output_bytes = dst_bytes;
    }
    // write data
    err = clEnqueueWriteBuffer(commands, mem_vy, CL_TRUE, 0, src1_bytes, vy, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(0);
    }
    // setup kernel
    err = 0;
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 0, sizeof(cl_mem), &mem_vx);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 1, sizeof(cl_mem), &mem_vy);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 2, sizeof(cl_mem), &mem_output);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 3, sizeof(unsigned int), &src0_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 4, sizeof(unsigned int), &src0_x_stride);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 5, sizeof(unsigned int), &src1_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 6, sizeof(unsigned int), &src1_x_stride);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 7, sizeof(unsigned int), &block_count);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 8, sizeof(unsigned int), &dst_x_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 9, sizeof(unsigned int), &dst_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8, 10, sizeof(unsigned int), &dst_count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(0);
    }
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    size_t global = dst_count;
    //size_t local = kernel_mul_mat_q4_q8_local;
    //if (local > global) local = global;
    //else global = (global + local - 1) / local * local;
    //err = clEnqueueNDRangeKernel(commands, kernel_mul_mat_q4_q8, 1, NULL, &global, &local, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(commands, kernel_mul_mat_q4_q8, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! \"%s\"\n", getErrorString(err));
        exit(0);
    }
    // Wait for the command commands to get serviced before reading back results
    err = clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: clFinish %d \"%s\"\n", err, getErrorString(err));
        exit(0);
    }
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(commands, mem_output, CL_TRUE, 0, dst_bytes, output, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d \"%s\"\n", err, getErrorString(err));
        exit(0);
    }

    // validate
    //mul_mat_q4_q8_validate(vx, vy, output, src0_y_width, src0_x_stride, src1_y_width, src1_x_stride, block_count, dst_x_width, dst_y_width, dst_count);
}


extern "C" void ocl_mul_mat_q4_q8_half_raw(
    block_q4_0 * vx,
    char* vyqs,
    ggml_fp16_t * vyd,
    float* output,
    const unsigned int src0_bytes,
    const unsigned int src1_bytes,
    const unsigned int dst_bytes,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    cl_int err;
    // allocate/reallocate scratch buffers if necessary
    cl_mem mem_vx;
    if (src0_cache) {
        if (!mem_vx_cache.count(vx)) {
            mem_vx = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_bytes, NULL, &err);
            err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_bytes, vx, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to write to source array!\n");
                exit(0);
            }
            mem_vx_cache[vx] = { mem_vx, NULL };
            mem_vx_cache_size += src0_bytes;
        }
        else {
            mem_vx = mem_vx_cache[vx].vx;
        }
    }
    else {
        if (mem_vx_bytes < src0_bytes) {
            if (mem_vx_bytes) {
                clReleaseMemObject(mem_vx_scratch);
            }
            mem_vx_scratch = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_bytes, NULL, &err);
            if (!mem_vx_scratch) {
                printf("Error: Failed to allocate device memory!\n");
                exit(0);
            }
            mem_vx_bytes = src0_bytes;
        }
        mem_vx = mem_vx_scratch;
        err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_bytes, vx, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(0);
        }
    }
    if (mem_vy_bytes < src1_bytes) {
        if (mem_vy_bytes) {
            clReleaseMemObject(mem_vy);
        }
        mem_vy = clCreateBuffer(context, CL_MEM_READ_ONLY, src1_bytes, NULL, &err);
        if (!mem_vy) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_vy_bytes = src1_bytes;
    }
    const unsigned int src1_d_bytes = src1_bytes * sizeof(ggml_fp16_t) / 32;
    if (mem_vyd_bytes < src1_d_bytes) {
        if (mem_vyd_bytes) {
            clReleaseMemObject(mem_vyd);
        }
        mem_vyd = clCreateBuffer(context, CL_MEM_READ_ONLY, src1_d_bytes, NULL, &err);
        if (!mem_vyd) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_vyd_bytes = src1_d_bytes;
    }
    if (mem_output_bytes < dst_bytes) {
        if (mem_output_bytes) {
            clReleaseMemObject(mem_output);
        }
        mem_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_bytes, NULL, &err);
        if (!mem_output) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_output_bytes = dst_bytes;
    }
    // write data
    err = clEnqueueWriteBuffer(commands, mem_vy, CL_FALSE, 0, src1_bytes, vyqs, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(0);
    }
    err = clEnqueueWriteBuffer(commands, mem_vyd, CL_FALSE, 0, src1_d_bytes, vyd, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(0);
    }
    // setup kernel
    err = 0;
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 0, sizeof(cl_mem), &mem_vx);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 1, sizeof(cl_mem), &mem_vy);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 2, sizeof(cl_mem), &mem_vyd);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 3, sizeof(cl_mem), &mem_output);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 4, sizeof(unsigned int), &src0_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 5, sizeof(unsigned int), &src0_x_stride);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 6, sizeof(unsigned int), &src1_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 7, sizeof(unsigned int), &src1_x_stride);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 8, sizeof(unsigned int), &block_count);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 9, sizeof(unsigned int), &dst_x_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 10, sizeof(unsigned int), &dst_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_half_raw, 11, sizeof(unsigned int), &dst_count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(0);
    }
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    size_t global = dst_count;
    err = clEnqueueNDRangeKernel(commands, kernel_mul_mat_q4_q8_half_raw, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! \"%s\"\n", getErrorString(err));
        exit(0);
    }
    // Wait for the command commands to get serviced before reading back results
    err = clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: clFinish %d \"%s\"\n", err, getErrorString(err));
        exit(0);
    }
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(commands, mem_output, CL_TRUE, 0, dst_bytes, output, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d \"%s\"\n", err, getErrorString(err));
        exit(0);
    }
}



extern "C" void ocl_mul_mat_q4_q8_raw(
    block_q4_0 * vx,
    char* vyqs,
    ggml_fp16_t * vyd,
    float* output,
    const unsigned int src0_bytes,
    const unsigned int src1_bytes,
    const unsigned int dst_bytes,
    const unsigned int src0_y_width,
    const unsigned int _src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    cl_int err;
    // allocate/reallocate scratch buffers if necessary
    unsigned int src0_x_stride = block_count * 16;
    cl_mem mem_vx, mem_vxd;
    if (src0_cache) {
        if (!mem_vx_cache.count(vx)) {
            const int count = src0_bytes / sizeof(block_q4_0);
            const int src0_qs_bytes = count * 16;
            const int src0_d_bytes = count * sizeof(vx->d);
            // extract blocks into raw arrays.
            uint8_t* vxqs = (uint8_t*)malloc(src0_qs_bytes + src0_d_bytes);
            ggml_fp16_t* vxd = (ggml_fp16_t*)(vxqs + src0_qs_bytes);
#ifdef USE_INTERLEAVE
            int qsi = 0, di = 0;
            uint8_t* qs = vxqs;
            ggml_fp16_t* d = vxd;
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < 16; j++) {
                    int x = qsi % src0_x_stride;
                    int y = qsi / src0_x_stride;
                    int iqsi = x * src0_y_width + y;
                    qs[iqsi] = vx[i].qs[j];
                    qsi++;
                }
                int x = di % (src0_x_stride / 16);
                int y = di / (src0_x_stride / 16);
                int idi = x * src0_y_width + y;
                d[idi] = vx[i].d;
                di++;
            }
#else
            uint8_t* qs = vxqs;
            ggml_fp16_t* d = vxd;
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < 16; j++) {
                    *(qs++) = vx[i].qs[j];
                }
                *(d++) = vx[i].d;
            }
#endif
            // write raw arrays
            mem_vx = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_qs_bytes, NULL, &err);
            err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_qs_bytes, vxqs, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to write to source array!\n");
                exit(0);
            }
            mem_vxd = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_d_bytes, NULL, &err);
            err = clEnqueueWriteBuffer(commands, mem_vxd, CL_TRUE, 0, src0_d_bytes, vxd, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to write to source array!\n");
                exit(0);
            }
            mem_vx_cache[vx] = { mem_vx, mem_vxd };
            mem_vx_cache_size += src0_qs_bytes + src0_d_bytes;
            free(vxqs);
        }
        else {
            vx_cache_t& cache = mem_vx_cache[vx];
            mem_vx = cache.vx;
            mem_vxd = cache.vxd;
        }
    }
    else {
        const int count = src0_bytes / sizeof(block_q4_0);
        const int src0_qs_bytes = count * 16;
        const int src0_d_bytes = count * sizeof(vx->d);
        // extract blocks into raw arrays.
        uint8_t* vxqs = (uint8_t*)malloc(src0_qs_bytes + src0_d_bytes);
        ggml_fp16_t* vxd = (ggml_fp16_t*)(vxqs + src0_qs_bytes);
        uint8_t* qs = vxqs;
        ggml_fp16_t* d = vxd;
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < 16; j++) {
                *(qs++) = vx[i].qs[j];
            }
            *(d++) = vx[i].d;
        }
        // qs
        if (mem_vx_bytes < src0_qs_bytes) {
            if (mem_vx_bytes) {
                clReleaseMemObject(mem_vx_scratch);
            }
            mem_vx_scratch = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_qs_bytes, NULL, &err);
            if (!mem_vx_scratch) {
                printf("Error: Failed to allocate device memory!\n");
                exit(0);
            }
            mem_vx_bytes = src0_qs_bytes;
        }
        mem_vx = mem_vx_scratch;
        err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_qs_bytes, vx, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(0);
        }
        // d
        if (mem_vxd_bytes < src0_d_bytes) {
            if (mem_vxd_bytes) {
                clReleaseMemObject(mem_vxd_scratch);
            }
            mem_vxd_scratch = clCreateBuffer(context, CL_MEM_READ_ONLY, src0_d_bytes, NULL, &err);
            if (!mem_vxd_scratch) {
                printf("Error: Failed to allocate device memory!\n");
                exit(0);
            }
            mem_vxd_bytes = src0_d_bytes;
        }
        mem_vxd = mem_vxd_scratch;
        err = clEnqueueWriteBuffer(commands, mem_vx, CL_TRUE, 0, src0_d_bytes, vx, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(0);
        }
        free(vxqs);
    }
    if (mem_vy_bytes < src1_bytes) {
        if (mem_vy_bytes) {
            clReleaseMemObject(mem_vy);
        }
        mem_vy = clCreateBuffer(context, CL_MEM_READ_ONLY, src1_bytes, NULL, &err);
        if (!mem_vy) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_vy_bytes = src1_bytes;
    }
    const unsigned int src1_d_bytes = src1_bytes * sizeof(ggml_fp16_t) / 32;
    if (mem_vyd_bytes < src1_d_bytes) {
        if (mem_vyd_bytes) {
            clReleaseMemObject(mem_vyd);
        }
        mem_vyd = clCreateBuffer(context, CL_MEM_READ_ONLY, src1_d_bytes, NULL, &err);
        if (!mem_vyd) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_vyd_bytes = src1_d_bytes;
    }
    if (mem_output_bytes < dst_bytes) {
        if (mem_output_bytes) {
            clReleaseMemObject(mem_output);
        }
        mem_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_bytes, NULL, &err);
        if (!mem_output) {
            printf("Error: Failed to allocate device memory!\n");
            exit(0);
        }
        mem_output_bytes = dst_bytes;
    }
    // write data
    err = clEnqueueWriteBuffer(commands, mem_vy, CL_FALSE, 0, src1_bytes, vyqs, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(0);
    }
    err = clEnqueueWriteBuffer(commands, mem_vyd, CL_FALSE, 0, src1_d_bytes, vyd, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(0);
    }
    // setup kernel
    err = 0;
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 0, sizeof(cl_mem), &mem_vx);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 1, sizeof(cl_mem), &mem_vxd);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 2, sizeof(cl_mem), &mem_vy);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 3, sizeof(cl_mem), &mem_vyd);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 4, sizeof(cl_mem), &mem_output);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 5, sizeof(unsigned int), &src0_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 6, sizeof(unsigned int), &src0_x_stride);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 7, sizeof(unsigned int), &src1_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 8, sizeof(unsigned int), &src1_x_stride);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 9, sizeof(unsigned int), &block_count);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 10, sizeof(unsigned int), &dst_x_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 11, sizeof(unsigned int), &dst_y_width);
    err |= clSetKernelArg(kernel_mul_mat_q4_q8_raw, 12, sizeof(unsigned int), &dst_count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(0);
    }
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    size_t global = dst_count;
    err = clEnqueueNDRangeKernel(commands, kernel_mul_mat_q4_q8_raw, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! \"%s\"\n", getErrorString(err));
        exit(0);
    }
    // Wait for the command commands to get serviced before reading back results
    err = clFinish(commands);
    if (err != CL_SUCCESS)
    {
        printf("Error: clFinish %d \"%s\"\n", err, getErrorString(err));
        exit(0);
    }
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(commands, mem_output, CL_TRUE, 0, dst_bytes, output, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d \"%s\"\n", err, getErrorString(err));
        exit(0);
    }
}
