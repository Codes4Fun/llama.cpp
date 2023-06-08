typedef unsigned char uint8_t;
typedef char int8_t;

#define QK4_0 32
typedef struct {
    half   d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
//static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, \"wrong q4_0 block size / padding\");

#define QK8_0 32
typedef struct {
    half   d;          // delta
    int8_t  qs[QK8_0];  // quants
} block_q8_0;
//static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, \"wrong q8_0 block size / padding\");

#if 0
float vec32_dot_q4_q8(
   __global block_q4_0 *x,
   __global block_q8_0 *y,
   const unsigned int nb)
{
    float fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        int sum = 0;
        for (int i = 0, i2 = 0; i < 16; i++, i2 += 2) {
            uint8_t q4 = x->qs[i];
            int x0 = (q4 & 0xf) - 8;
            int x1 = (q4 >> 4) - 8;
            int y0 = y->qs[i2];
            int y1 = y->qs[i2+1];
            sum += x0*y0 + x1*y1;
        }
        fsum += sum * vload_half(0,&x->d) * vload_half(0,&y->d);
        x++; y++;
    }
    return fsum;
}
#endif

float vec32_dot_q4_q8_b(
   __global block_q4_0 *x,
   __global block_q8_0 *y,
   const unsigned int nb)
{
    float fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        int sum = 0;
        __global uint8_t *xqs = x->qs;
        __global int8_t *yqs0 = y->qs;
        __global int8_t *yqs1 = y->qs + (QK8_0/2);
        for (int i = 0; i < 16; i++) {
            uint8_t q4 = *(xqs++);
            int x0 = (q4 & 0xf) - 8;
            int x1 = (q4 >> 4) - 8;
            int y0 = *(yqs0++);
            int y1 = *(yqs1++);
            sum += x0*y0 + x1*y1;
        }
        fsum += sum * (float)vload_half(0,&x->d) * (float)vload_half(0,&y->d);
        x++; y++;
    }
    return fsum;
}

float vec32_dot_q4_q8_c(
   __global block_q4_0 *x,
   __global block_q8_0 *y,
   unsigned int nb)
{
    float fsum = 0;
    for (; nb >= 4; nb-=4) {
        float d[4];
        int q[4];
        for (int j = 0; j < 4; j++) {
            d[j] = vload_half(0,&x->d) * vload_half(0,&y->d);
            __global uint8_t *xqs = x->qs;
            __global int8_t *yqs = y->qs;
            int sum = 0;
            for (int i = 0; i < 2; i++) {
                short16 _xqs, _yqs;
                uchar8 _xqs4 = vload8(i,xqs);
                _xqs.even = convert_short8(_xqs4 & (uchar8)(0xf));
                _xqs.odd = convert_short8(_xqs4 >> (uchar8)(4));
                _xqs -= (short16)(8);
                _yqs.even = convert_short8(vload8(i,yqs));
                _yqs.odd = convert_short8(vload8(i+2,yqs));
                short16 qs16 = _xqs * _yqs;
                short8 qs8 = qs16.s01234567 + qs16.s89abcdef;
                short4 qs4 = qs8.s0123 + qs8.s4567;
                short2 qs = qs4.s01 + qs4.s23;
                sum += qs.s0 + qs.s1;
            }
            q[j] = sum;
            x++; y++;
        }
        for (int j = 0; j < 4; j++) {
            fsum += q[j] * d[j];
        }
    }
    for (; nb > 0; nb--) {
        float d = vload_half(0,&x->d) * vload_half(0,&y->d);
        __global uint8_t *xqs = x->qs;
        __global int8_t *yqs = y->qs;
        int sum = 0;
        for (int i = 0; i < 2; i++) {
            short16 _xqs, _yqs;
            uchar8 _xqs4 = vload8(i,xqs);
            _xqs.even = convert_short8(_xqs4 & (uchar8)(0xf));
            _xqs.odd = convert_short8(_xqs4 >> (uchar8)(4));
            _xqs -= (short16)(8);
            _yqs.even = convert_short8(vload8(i,yqs));
            _yqs.odd = convert_short8(vload8(i+2,yqs));
            short16 qs16 = _xqs * _yqs;
            short8 qs8 = qs16.s01234567 + qs16.s89abcdef;
            short4 qs4 = qs8.s0123 + qs8.s4567;
            short2 qs = qs4.s01 + qs4.s23;
            sum += qs.s0 + qs.s1;
        }
        fsum += sum * d;//vload_half(0,&x->d) * vload_half(0,&y->d);
        x++; y++;
    }
    return fsum;
}

__kernel void kernel_mul_mat_q4_q8(
   __global block_q4_0 *vx,
   __global block_q8_0 *vy,
   __global float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    int k = get_global_id(0);
    if(k < dst_count) {
        const int x = k % dst_x_width;
        const int yzw = k / dst_x_width;
        const int y = yzw % dst_y_width;
        const int zw = yzw / dst_y_width;
        const int src0_index = x + src0_y_width * zw;
        __global block_q4_0* src0_data = (__global block_q4_0*)((__global char*)vx + (src0_index * src0_x_stride));
        const int src1_index = y + src1_y_width * zw;
        __global block_q8_0* src1_data = (__global block_q8_0*)((__global char*)vy + (src1_index * src1_x_stride));
        output[k] = vec32_dot_q4_q8_c(src0_data, src1_data, block_count);
    }
}

float vec32_dot_q4_q8_half_raw(
   __global block_q4_0 *x,
   __global char *yqs,
   __global half *yd,
   const unsigned int nb)
{
    float fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        int sum = 0;
        __global uint8_t *xqs = x->qs;
        for (int i = 0; i < 16; i++) {
            uint8_t q4 = *xqs;
            int x0 = (q4 & 0xf) - 8;
            int x1 = (q4 >> 4) - 8;
            int y0 = *(yqs++);
            int y1 = *yqs;
            sum += x0*y0 + x1*y1;
            xqs++; yqs++;
        }
        fsum += sum * vload_half(0,&x->d) * vload_half(0,yd);
        x++; yd++;
    }
    return fsum;
}

__kernel void kernel_mul_mat_q4_q8_half_raw(
   __global block_q4_0 *vx,
   __global char *vyqs,
   __global half *vyd,
   __global float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    int k = get_global_id(0);
    if(k < dst_count) {
        const int x = k % dst_x_width;
        const int yzw = k / dst_x_width;
        const int y = yzw % dst_y_width;
        const int zw = yzw / dst_y_width;
        const int src0_index = x + src0_y_width * zw;
        const int src1_index = y + src1_y_width * zw;
        __global block_q4_0* src0_data = (__global block_q4_0*)((__global char*)vx + (src0_index * src0_x_stride));
        __global char* src1_qs = (vyqs + (src1_index * src1_x_stride));
        __global half* src1_d = (vyd + (src1_index * src1_x_stride / 32));
        output[k] = vec32_dot_q4_q8_half_raw(src0_data, src1_qs, src1_d, block_count);
    }
}

float vec32_dot_q4_q8_raw(
   __global uchar *xqs,
   __global half *xd,
   __global char *yqs,
   __global half *yd,
   const unsigned int nb)
{
    float fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) {
            uint8_t q4 = *xqs;
            int x0 = (q4 & 0xf) - 8;
            int x1 = (q4 >> 4) - 8;
            int y0 = *(yqs++);
            int y1 = *yqs;
            sum += x0*y0 + x1*y1;
            xqs++; yqs++;
        }
        fsum += sum * vload_half(0,xd) * vload_half(0,yd);
        xd++; yd++;
    }
    return fsum;
}

__kernel void kernel_mul_mat_q4_q8_raw(
   __global uchar *vxqs,
   __global half *vxd,
   __global char *vyqs,
   __global half *vyd,
   __global float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    int k = get_global_id(0);
    if(k < dst_count) {
        const int x = k % dst_x_width;
        const int yzw = k / dst_x_width;
        const int y = yzw % dst_y_width;
        const int zw = yzw / dst_y_width;
        const int src0_index = (x + src0_y_width * zw);
        const int src1_index = (y + src1_y_width * zw);
        __global uchar* src0_qs = (vxqs + (src0_index * src0_x_stride));
        __global half* src0_d = (vxd + (src0_index * src0_x_stride / 16));
        __global char* src1_qs = (vyqs + (src1_index * src1_x_stride));
        __global half* src1_d = (vyd + (src1_index * src1_x_stride / 32));
        output[k] = vec32_dot_q4_q8_raw(src0_qs, src0_d, src1_qs, src1_d, block_count);
    }
}



float vec32_dot_q4_q8_raw8(
   __global uchar4 *xqs,
   __global half *xd,
   __global char8 *yqs,
   __global half *yd,
   const unsigned int nb)
{
    float8 fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        float8 sum = 0;
        for (int i = 0; i < 4; i++) {
            uchar4 q4 = *xqs;
            float8 _xqs;
            _xqs.even = convert_float4(q4 & (uchar4)(0xf));
            _xqs.odd = convert_float4(q4 >> (uchar4)(4));
            _xqs -= (float8)(8);
            _xqs *= convert_float8(*yqs);
            sum += _xqs;
            xqs++; yqs++;
        }
        fsum += sum * vload_half(0,xd) * vload_half(0,yd);
        xd++; yd++;
    }
    fsum.s0123 += fsum.s4567;
    fsum.s01 += fsum.s23;
    return fsum.s0 + fsum.s1;
}

__kernel void kernel_mul_mat_q4_q8_raw8(
   __global uchar4 *vxqs,
   __global half *vxd,
   __global char8 *vyqs,
   __global half *vyd,
   __global float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    int k = get_global_id(0);
    if(k < dst_count) {
        const int x = k % dst_x_width;
        const int yzw = k / dst_x_width;
        const int y = yzw % dst_y_width;
        const int zw = yzw / dst_y_width;
        const int src0_index = (x + src0_y_width * zw);
        const int src1_index = (y + src1_y_width * zw);
        __global uchar4* src0_qs = (vxqs + (src0_index * src0_x_stride / 4));
        __global half* src0_d = (vxd + (src0_index * src0_x_stride / 16));
        __global char8* src1_qs = (vyqs + (src1_index * src1_x_stride / 8));
        __global half* src1_d = (vyd + (src1_index * src1_x_stride / 32));
        output[k] = vec32_dot_q4_q8_raw8(src0_qs, src0_d, src1_qs, src1_d, block_count);
    }
}




float vec32_dot_q4_q8_interleave(
   __global uchar *xqs,
   __global half *xd,
   const uint xstride,
   __global char2 *yqs,
   __global half *yd,
   const unsigned int nb)
{
    float fsum = 0;
    for (unsigned int j = 0; j < nb; j++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) {
            uchar q4 = *xqs;
            int2 x = (int2)((q4 & 0xf), (q4 >> 4)) - (int2)(8);
            int2 y = convert_int2(*yqs);
            x *= y;
            sum += x.x + x.y;
            xqs += xstride;
            yqs++;
        }
        fsum += sum * vload_half(0,xd) * vload_half(0,yd);
        xd += xstride;
        yd++;
    }
    return fsum;
}

__kernel void kernel_mul_mat_q4_q8_interleave(
   __global uchar *vxqs,
   __global half *vxd,
   __global char2 *vyqs,
   __global half *vyd,
   __global float* output,
    const unsigned int src0_y_width,
    const unsigned int src0_x_stride,
    const unsigned int src1_y_width,
    const unsigned int src1_x_stride,
    const unsigned int block_count,
    const unsigned int dst_x_width,
    const unsigned int dst_y_width,
    const unsigned int dst_count
) {
    int k = get_global_id(0);
    if(k < dst_count) {
        const int x = k % dst_x_width;
        const int yzw = k / dst_x_width;
        const int y = yzw % dst_y_width;
        const int zw = yzw / dst_y_width;
        const int src0_index = (x + src0_y_width * zw);
        const int src1_index = (y + src1_y_width * zw);
        __global uchar* src0_qs = (vxqs + (src0_index));// * src0_x_stride));
        __global half* src0_d = (vxd + (src0_index));// * src0_x_stride / 16));
        __global char2* src1_qs = (vyqs + (src1_index * src1_x_stride / 2));
        __global half* src1_d = (vyd + (src1_index * src1_x_stride / 32));
        output[k] = vec32_dot_q4_q8_interleave(
            src0_qs, src0_d, src0_y_width,
            src1_qs, src1_d,
            block_count
        );
    }
}

