#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>

namespace StyleTransfer{

typedef unsigned short half;

Inference :: Inference(std::string offline_model){
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT){
    // TODO:load model
    cnrtInit(0);
    cnrtModel_t model;
    cnrtLoadModel(&model, offline_model_.c_str());
    // TODO:set current device
    unsigned dev_num;
    cnrtGetDeviceCount(&dev_num);
    if (dev_num == 0) {
        return;
    }
    cnrtDev_t device;
    cnrtGetDeviceHandle(&device, 0);
    cnrtSetCurrentDevice(device);

    // TODO:load extract function
    int number = 0;
    cnrtGetFunctionNumber(model, &number);
    printf("%d function\n", number);
    cnrtFunction_t function;
    if (cnrtCreateFunction(&function) != CNRT_RET_SUCCESS) {
        printf("cnrtCreateFunction Failed!\n");
        exit(-1);
    }
    if (cnrtExtractFunction(&function, model, "subnet0") != CNRT_RET_SUCCESS) {
        printf("cnrtExtractFunction Failed!\n");
        exit(-1);
    }
    // TODO:prepare data on cpu
    // 完成NCHW到NHWC的转换
    float* inputs = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    float* outputs = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    int size = 256*256; // H*W
    int channels = 3; // C
    // 这个转换妙啊!
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < channels; j++) {
            inputs[i*channels+j] = DataT->input_data[size*j+i];
        }
    }

    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
    printf("inputNum: %d, outputNum: %d", inputNum, outputNum);
    // TODO:allocate I/O data memory on MLU
    void *mlu_input, *mlu_output;

    // TODO:malloc cpu memory
    half* input_half = (half*)malloc(256*256*3*sizeof(half));
    half* output_half = (half*)malloc(256*256*3*sizeof(half));
    // TODO:malloc mlu memory
    DataT->output_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    
    for (int i = 0; i < size*channels; i++) {
        cnrtConvertFloatToHalf(input_half+i, inputs[i]);
    }
    for (int i = 0; i < size*channels; i++) {
        cnrtConvertFloatToHalf(input_half+i, inputs[i]);
    }
    // TODO:prepare input/output buffer
    if (cnrtMalloc(&(mlu_input), inputSizeS[0]) != CNRT_RET_SUCCESS) {
        printf("cnrtMalloc input Failed!\n");
        exit(-1);
    }
    if (cnrtMalloc(&(mlu_output), outputSizeS[0]) != CNRT_RET_SUCCESS) {
        printf("cnrtMalloc output Failed!\n");
        exit(-1);
    }
    if (cnrtMemcpy(mlu_input, input_half, 256*256*3*sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV) != CNRT_RET_SUCCESS) {
        printf("cnrtMemcpy input Failed!\n");
        exit(-1);
    }

    // TODO:setup runtime ctx
    cnrtRuntimeContext_t ctx;
    cnrtCreateRuntimeContext(&ctx, function, NULL);
    // TODO:bind device
    cnrtSetRuntimeContextDeviceId(ctx, 0);
    cnrtInitRuntimeContext(ctx, NULL);
    
    void *param[2];
    param[0] = mlu_input;
    param[1] = mlu_output;
    // TODO:compute offline
    cnrtQueue_t queue;
    cnrtRuntimeContextCreateQueue(ctx, &queue);
    cnrtInvokeRuntimeContext(ctx, (void**)param, queue, NULL);
    cnrtSyncQueue(queue);
    printf("run success!\n");

    if (cnrtMemcpy(output_half, mlu_output, 256*256*3*sizeof(half), CNRT_MEM_TRANS_DIR_DEV2HOST) != CNRT_RET_SUCCESS) {
        printf("cnrtMemcpy output Failed!\n");
        exit(-1);
    }
    for (int i = 0; i < size*channels; i++) {
        cnrtConvertHalfToFloat(outputs+i, output_half[i]);
    }
    printf("memcpy output success!\n");

    for(int i = 0; i < size; i++) {
        for (int j = 0 ; j < channels; j++) {
            DataT->output_data[size*j+i] = outputs[i*channels+j];
        }
    }

    // TODO:free memory spac
    if (cnrtFree(mlu_input) != CNRT_RET_SUCCESS) {
        printf("cnrtFree input Failed.\n");
        exit(-1);
    }
    printf("free mlu success!\n");
    if (cnrtDestroyQueue(queue) != CNRT_RET_SUCCESS) {
        printf("cnrtDestroyQueue Failed!\n");
        exit(-1);
    }
    printf("destroy queue success!\n");
    cnrtDestroy();
    free(input_half);
    free(output_half);
    // free(inputs);
    // free(outputs);
}

} // namespace StyleTransfer
