/***************************************************************************
 # Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RESTIR_PARAMETERS_H
#define RESTIR_PARAMETERS_H

#include "RestirgdiTypes.h"

// Flag that is used in the RIS buffer to identify that a light is 
// stored in a compact form.
#define RESTIR_LIGHT_COMPACT_BIT 0x80000000u

// Light index mask for the RIS buffer.
#define RESTIR_LIGHT_INDEX_MASK 0x7fffffff

// Reservoirs are stored in a structured buffer in a block-linear layout.
// This constant defines the size of that block, measured in pixels.
#define RESTIR_RESERVOIR_BLOCK_SIZE 16

// Bias correction modes for temporal and spatial resampling:
// Use (1/M) normalization, which is very biased but also very fast.
#define RESTIR_BIAS_CORRECTION_OFF 0
// Use MIS-like normalization but assume that every sample is visible.
#define RESTIR_BIAS_CORRECTION_BASIC 1
// Use pairwise MIS normalization (assuming every sample is visible).  Better perf & specular quality
#define RESTIR_BIAS_CORRECTION_PAIRWISE 2
// Use MIS-like normalization with visibility rays. Unbiased.
#define RESTIR_BIAS_CORRECTION_RAY_TRACED 3


#define RESTIR_ONION_MAX_LAYER_GROUPS 8
#define RESTIR_ONION_MAX_RINGS 52

#define RESTIR_INVALID_LIGHT_INDEX (0xffffffffu)

#ifndef __cplusplus
static const uint RESTIR_InvalidLightIndex = RESTIR_INVALID_LIGHT_INDEX;
#endif

struct RESTIR_OnionLayerGroup
{
    float innerRadius;
    float outerRadius;
    float invLogLayerScale;
    int layerCount;

    float invEquatorialCellAngle;
    int cellsPerLayer;
    int ringOffset;
    int ringCount;

    float equatorialCellAngle;
    float layerScale;
    int layerCellOffset;
    int pad;
};

struct RESTIR_OnionRing
{
    float cellAngle;
    float invCellAngle;
    int cellOffset;
    int cellCount;
};

struct RESTIR_ResamplingRuntimeParameters
{
    uint32_t firstLocalLight;
    uint32_t numLocalLights;
    uint32_t firstInfiniteLight;
    uint32_t numInfiniteLights;

    uint32_t environmentLightPresent;
    uint32_t environmentLightIndex;
    uint32_t tileSize;
    uint32_t tileCount;

    uint32_t activeCheckerboardField; // 0 - no checkerboard, 1 - odd pixels, 2 - even pixels
    uint32_t enableLocalLightImportanceSampling;
    uint32_t reservoirBlockRowPitch;
    uint32_t reservoirArrayPitch;

    uint32_t environmentRisBufferOffset;
    uint32_t environmentTileSize;
    uint32_t environmentTileCount;
    uint32_t neighborOffsetMask;

    uint32_t uniformRandomNumber;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};

struct RESTIR_PackedReservoir
{
    uint32_t lightData;
    uint32_t uvData;
    uint32_t mVisibility;
    uint32_t distanceAge;
    float targetPdf;
    float weight;
};

#endif // RESTIR_PARAMETERS_H
