#ifndef CL_COMMON_H
#define CL_COMMON_H

// OpenCL includes
#include <CL/cl.h>

extern cl_uint numPlatforms;
extern cl_platform_id* platform_ids;
extern cl_uint numDevices;
extern cl_device_id* device_ids;
extern char device_name[4096];
extern size_t device_name_length;
extern cl_context context;
extern cl_command_queue commands;

#ifdef __cplusplus
extern "C" {
#endif

int ocl_context_init();
int ocl_context_release();

#ifdef __cplusplus
}
#endif

const char* getErrorString(cl_int error);

#endif//CL_COMMON_H
