/***************************************************************************
 # Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RESERVOIR_SLANG
#define RESERVOIR_SLANG

#include "RestirgdiParameters.h"
#include "RestirgdiHelpers.slang"

#ifndef RESTIR_LIGHT_RESERVOIR_BUFFER
#error "RESTIR_LIGHT_RESERVOIR_BUFFER must be defined to point to a RWStructuredBuffer<RESTIR_PackedReservoir> type resource"
#endif

// Define this macro to 0 if your shader needs read-only access to the reservoirs, 
// to avoid compile errors in the RESTIR_StoreReservoir function
#ifndef RESTIR_ENABLE_STORE_RESERVOIR
#define RESTIR_ENABLE_STORE_RESERVOIR 1
#endif

// This structure represents a single light reservoir that stores the weights, the sample ref,
// sample count (M), and visibility for reuse. It can be serialized into RESTIR_PackedReservoir for storage.
struct RESTIR_Reservoir
{
    // Light index (bits 0..30) and validity bit (31)
    uint lightData;     

    // Sample UV encoded in 16-bit fixed point format
    uint uvData;        

    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;

    // Target PDF of the selected sample
    float targetPdf;

    // Number of samples considered for this reservoir (pairwise MIS makes this a float)
    float M;

    // Visibility information stored in the reservoir for reuse
    uint packedVisibility;

    // Screen-space distance between the current location of the reservoir
    // and the location where the visibility information was generated,
    // minus the motion vectors applied in temporal resampling
    int2 spatialDistance;

    // How many frames ago the visibility information was generated
    uint age;

    // Cannonical weight when using pairwise MIS (ignored except during pairwise MIS computations)
    float canonicalWeight;
};

// Encoding helper constants for RESTIR_PackedReservoir.mVisibility
static const uint RESTIR_PackedReservoir_VisibilityMask = 0x3ffff;
static const uint RESTIR_PackedReservoir_VisibilityChannelMax = 0x3f;
static const uint RESTIR_PackedReservoir_VisibilityChannelShift = 6;
static const uint RESTIR_PackedReservoir_MShift = 18;
static const uint RESTIR_PackedReservoir_MaxM = 0x3fff;

// Encoding helper constants for RESTIR_PackedReservoir.distanceAge
static const uint RESTIR_PackedReservoir_DistanceChannelBits = 8;
static const uint RESTIR_PackedReservoir_DistanceXShift = 0;
static const uint RESTIR_PackedReservoir_DistanceYShift = 8;
static const uint RESTIR_PackedReservoir_AgeShift = 16;
static const uint RESTIR_PackedReservoir_MaxAge = 0xff;
static const uint RESTIR_PackedReservoir_DistanceMask = (1u << RESTIR_PackedReservoir_DistanceChannelBits) - 1;
static const  int RESTIR_PackedReservoir_MaxDistance = int((1u << (RESTIR_PackedReservoir_DistanceChannelBits - 1)) - 1);

// Light index helpers
static const uint RESTIR_Reservoir_LightValidBit = 0x80000000;
static const uint RESTIR_Reservoir_LightIndexMask = 0x7FFFFFFF;

RESTIR_PackedReservoir RESTIR_PackReservoir(const RESTIR_Reservoir reservoir)
{
    int2 clampedSpatialDistance = clamp(reservoir.spatialDistance, -RESTIR_PackedReservoir_MaxDistance, RESTIR_PackedReservoir_MaxDistance);
    uint clampedAge = clamp(reservoir.age, 0, RESTIR_PackedReservoir_MaxAge);

    RESTIR_PackedReservoir data;
    data.lightData = reservoir.lightData;
    data.uvData = reservoir.uvData;

    data.mVisibility = reservoir.packedVisibility
        | (min(uint(reservoir.M), RESTIR_PackedReservoir_MaxM) << RESTIR_PackedReservoir_MShift);

    data.distanceAge = 
          ((clampedSpatialDistance.x & RESTIR_PackedReservoir_DistanceMask) << RESTIR_PackedReservoir_DistanceXShift) 
        | ((clampedSpatialDistance.y & RESTIR_PackedReservoir_DistanceMask) << RESTIR_PackedReservoir_DistanceYShift) 
        | (clampedAge << RESTIR_PackedReservoir_AgeShift);

    data.targetPdf = reservoir.targetPdf;
    data.weight = reservoir.weightSum;

    return data;
}

#if RESTIR_ENABLE_STORE_RESERVOIR
void RESTIR_StoreReservoir(
    const RESTIR_Reservoir reservoir,
    RESTIR_ResamplingRuntimeParameters params,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RESTIR_ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    RESTIR_LIGHT_RESERVOIR_BUFFER[pointer] = RESTIR_PackReservoir(reservoir);
}
#endif // RESTIR_ENABLE_STORE_RESERVOIR

RESTIR_Reservoir RESTIR_EmptyReservoir()
{
    RESTIR_Reservoir s;
    s.lightData = 0;
    s.uvData = 0;
    s.targetPdf = 0;
    s.weightSum = 0;
    s.M = 0;
    s.packedVisibility = 0;
    s.spatialDistance = int2(0, 0);
    s.age = 0;
    s.canonicalWeight = 0;
    return s;
}

RESTIR_Reservoir RESTIR_UnpackReservoir(RESTIR_PackedReservoir data)
{
    RESTIR_Reservoir res;
    res.lightData = data.lightData;
    res.uvData = data.uvData;
    res.targetPdf = data.targetPdf;
    res.weightSum = data.weight;
    res.M = (data.mVisibility >> RESTIR_PackedReservoir_MShift) & RESTIR_PackedReservoir_MaxM;
    res.packedVisibility = data.mVisibility & RESTIR_PackedReservoir_VisibilityMask;
    // Sign extend the shift values
    res.spatialDistance.x = int(data.distanceAge << (32 - RESTIR_PackedReservoir_DistanceXShift - RESTIR_PackedReservoir_DistanceChannelBits)) >> (32 - RESTIR_PackedReservoir_DistanceChannelBits);
    res.spatialDistance.y = int(data.distanceAge << (32 - RESTIR_PackedReservoir_DistanceYShift - RESTIR_PackedReservoir_DistanceChannelBits)) >> (32 - RESTIR_PackedReservoir_DistanceChannelBits);
    res.age = (data.distanceAge >> RESTIR_PackedReservoir_AgeShift) & RESTIR_PackedReservoir_MaxAge;
    res.canonicalWeight = 0.0f;

    // Discard reservoirs that have Inf/NaN
    if (isinf(res.weightSum) || isnan(res.weightSum)) {
        res = RESTIR_EmptyReservoir();
    }

    return res;
}

RESTIR_Reservoir RESTIR_LoadReservoir(
    RESTIR_ResamplingRuntimeParameters params,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RESTIR_ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    return RESTIR_UnpackReservoir(RESTIR_LIGHT_RESERVOIR_BUFFER[pointer]);
}

void RESTIR_StoreVisibilityInReservoir(
    inout RESTIR_Reservoir reservoir,
    float3 visibility,
    bool discardIfInvisible)
{
    reservoir.packedVisibility = uint(saturate(visibility.x) * RESTIR_PackedReservoir_VisibilityChannelMax) 
        | (uint(saturate(visibility.y) * RESTIR_PackedReservoir_VisibilityChannelMax)) << RESTIR_PackedReservoir_VisibilityChannelShift
        | (uint(saturate(visibility.z) * RESTIR_PackedReservoir_VisibilityChannelMax)) << (RESTIR_PackedReservoir_VisibilityChannelShift * 2);

    reservoir.spatialDistance = int2(0, 0);
    reservoir.age = 0;

    if (discardIfInvisible && visibility.x == 0 && visibility.y == 0 && visibility.z == 0)
    {
        // Keep M for correct resampling, remove the actual sample
        reservoir.lightData = 0;
        reservoir.weightSum = 0;
    }
}

// Structure that groups the parameters for RESTIR_GetReservoirVisibility(...)
// Reusing final visibility reduces the number of high-quality shadow rays needed to shade
// the scene, at the cost of somewhat softer or laggier shadows.
struct RESTIR_VisibilityReuseParameters
{
    // Controls the maximum age of the final visibility term, measured in frames, that can be reused from the
    // previous frame(s). Higher values result in better performance.
    uint maxAge;

    // Controls the maximum distance in screen space between the current pixel and the pixel that has
    // produced the final visibility term. The distance does not include the motion vectors.
    // Higher values result in better performance and softer shadows.
    float maxDistance;
};

bool RESTIR_GetReservoirVisibility(
    const RESTIR_Reservoir reservoir,
    const RESTIR_VisibilityReuseParameters params,
    out float3 o_visibility)
{
    if (reservoir.age > 0 &&
        reservoir.age <= params.maxAge &&
        length(float2(reservoir.spatialDistance)) < params.maxDistance)
    {
        o_visibility.x = float(reservoir.packedVisibility & RESTIR_PackedReservoir_VisibilityChannelMax) / RESTIR_PackedReservoir_VisibilityChannelMax;
        o_visibility.y = float((reservoir.packedVisibility >> RESTIR_PackedReservoir_VisibilityChannelShift) & RESTIR_PackedReservoir_VisibilityChannelMax) / RESTIR_PackedReservoir_VisibilityChannelMax;
        o_visibility.z = float((reservoir.packedVisibility >> (RESTIR_PackedReservoir_VisibilityChannelShift * 2)) & RESTIR_PackedReservoir_VisibilityChannelMax) / RESTIR_PackedReservoir_VisibilityChannelMax;

        return true;
    }

    o_visibility = float3(0, 0, 0);
    return false;
}

bool RESTIR_IsValidReservoir(const RESTIR_Reservoir reservoir)
{
    return reservoir.lightData != 0;
}

uint RESTIR_GetReservoirLightIndex(const RESTIR_Reservoir reservoir)
{
    return reservoir.lightData & RESTIR_Reservoir_LightIndexMask;
}

float2 RESTIR_GetReservoirSampleUV(const RESTIR_Reservoir reservoir)
{
    return float2(reservoir.uvData & 0xffff, reservoir.uvData >> 16) / float(0xffff);
}

float RESTIR_GetReservoirInvPdf(const RESTIR_Reservoir reservoir)
{
    return reservoir.weightSum;
}

bool RESTIR_IsActiveCheckerboardPixel(
    uint2 pixelPosition,
    bool previousFrame,
    RESTIR_ResamplingRuntimeParameters params)
{
    if (params.activeCheckerboardField == 0)
        return true;

    return ((pixelPosition.x + pixelPosition.y + int(previousFrame)) & 1) == (params.activeCheckerboardField & 1);
}

uint2 RESTIR_PixelPosToReservoir(uint2 pixelPosition, RESTIR_ResamplingRuntimeParameters params)
{
    if (params.activeCheckerboardField == 0)
        return pixelPosition;

    return uint2(pixelPosition.x >> 1, pixelPosition.y);
}

uint2 RESTIR_ReservoirToPixelPos(uint2 reservoirIndex, RESTIR_ResamplingRuntimeParameters params)
{
    if (params.activeCheckerboardField == 0)
        return reservoirIndex;

    uint2 pixelPosition = uint2(reservoirIndex.x << 1, reservoirIndex.y);
    pixelPosition.x += ((pixelPosition.y + params.activeCheckerboardField) & 1);
    return pixelPosition;
}

#endif // RESERVOIR_SLANG
