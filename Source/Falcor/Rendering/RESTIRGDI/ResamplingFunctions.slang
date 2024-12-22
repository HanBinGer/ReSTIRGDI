/***************************************************************************
 # Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RESAMPLING_FUNCTIONS_SLANG
#define RESAMPLING_FUNCTIONS_SLANG

#include "Reservoir.slang"

// This macro can be defined in the including shader file to reduce code bloat
// and/or remove ray tracing calls from temporal and spatial resampling shaders
// if bias correction is not necessary.
#ifndef RESTIR_ALLOWED_BIAS_CORRECTION
#define RESTIR_ALLOWED_BIAS_CORRECTION RESTIR_BIAS_CORRECTION_RAY_TRACED
#endif

// This macro enables the functions that deal with the RIS buffer and presampling.
#ifndef RESTIR_ENABLE_PRESAMPLING
#define RESTIR_ENABLE_PRESAMPLING 1
#endif

#if RESTIR_ENABLE_PRESAMPLING && !defined(RESTIR_RIS_BUFFER)
#error "RESTIR_RIS_BUFFER must be defined to point to a RWBuffer<uint2> type resource"
#endif

#ifndef RESTIR_NEIGHBOR_OFFSETS_BUFFER
#error "RESTIR_NEIGHBOR_OFFSETS_BUFFER must be defined to point to a Buffer<float2> type resource"
#endif

struct RESTIR_SampleParameters
{
    uint numRegirSamples;
    uint numLocalLightSamples;
    uint numInfiniteLightSamples;
    uint numEnvironmentMapSamples;
    uint numBrdfSamples;

    uint numMisSamples;
    float localLightMisWeight;
    float environmentMapMisWeight;
    float brdfMisWeight;
    float brdfCutoff; 
    float brdfRayMinT;
};

// Sample parameters struct
// Defined so that so these can be compile time constants as defined by the user
// brdfCutoff Value in range [0,1] to determine how much to shorten BRDF rays. 0 to disable shortening
RESTIR_SampleParameters RESTIR_InitSampleParameters(
    uint numRegirSamples,
    uint numLocalLightSamples,
    uint numInfiniteLightSamples,
    uint numEnvironmentMapSamples,
    uint numBrdfSamples,
    float brdfCutoff RESTIR_DEFAULT(0.0),
    float brdfRayMinT RESTIR_DEFAULT(0.001f))
{
    RESTIR_SampleParameters result;
    result.numRegirSamples = numRegirSamples;
    result.numLocalLightSamples = numLocalLightSamples;
    result.numInfiniteLightSamples = numInfiniteLightSamples;
    result.numEnvironmentMapSamples = numEnvironmentMapSamples;
    result.numBrdfSamples = numBrdfSamples;

    result.numMisSamples = numLocalLightSamples + numEnvironmentMapSamples + numBrdfSamples;
    result.localLightMisWeight = float(numLocalLightSamples) / result.numMisSamples;
    result.environmentMapMisWeight = float(numEnvironmentMapSamples) / result.numMisSamples;
    result.brdfMisWeight = float(numBrdfSamples) / result.numMisSamples;
    result.brdfCutoff = brdfCutoff;
    result.brdfRayMinT = brdfRayMinT;

    return result;
}

/** Heuristic to determine a max visibility ray length from a PDF wrt. solid angle.
       \param[in] pdf PDF wrt. solid angle.
   */
float RESTIR_BrdfMaxDistanceFromPdf(float brdfCutoff, float pdf)
{
    const float kRayTMax = 3.402823466e+38F; // FLT_MAX
    return brdfCutoff > 0.f ? sqrt((1.f / brdfCutoff - 1.f) * pdf) : kRayTMax;
}

/** Computes the multi importance sampling pdf for brdf and light sample.
    For light and BRDF PDFs wrt solid angle, blend between the two
        \param[in] lightSelectionPdf is a dimensionless selection pdf
*/
float RESTIR_LightBrdfMisWeight(RAB_Surface surface, RAB_LightSample lightSample, 
    float lightSelectionPdf, float lightMisWeight, bool isEnvironmentMap,
    RESTIR_SampleParameters sampleParams)
{
    float lightSolidAnglePdf = RAB_LightSampleSolidAnglePdf(lightSample);
    if (sampleParams.brdfMisWeight == 0 || RAB_IsAnalyticLightSample(lightSample) || 
        lightSolidAnglePdf <= 0 || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
    {
        // BRDF samples disabled or we can't trace BRDF rays MIS with analytical lights
        return lightMisWeight * lightSelectionPdf;
    }

    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

    // Compensate for ray shortening due to brdf cutoff, does not apply to environment map sampling
    float brdfPdf = RAB_GetSurfaceBrdfPdf(surface, lightDir);
    float maxDistance = RESTIR_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
    if (!isEnvironmentMap && lightDistance > maxDistance)
        brdfPdf = 0.f;

    // Convert light selection pdf (unitless) to a solid angle measurement
    float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

    // MIS blending against solid angle pdfs.
    float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + sampleParams.brdfMisWeight * brdfPdf;

    // Convert back, RESTIR divides shading again by this term later
    return blendedPdfWrtSolidangle / lightSolidAnglePdf;
}

// Adds a new, non-reservoir light sample into the reservoir, returns true if this sample was selected.
// Algorithm (3) from the ReSTIR paper, Streaming RIS using weighted reservoir sampling.
bool RESTIR_StreamSample(
    inout RESTIR_Reservoir reservoir,
    uint lightIndex,
    float2 uv,
    float random,
    float targetPdf,
    float invSourcePdf)
{
    // What's the current weight
    float risWeight = targetPdf * invSourcePdf;

    // Add one sample to the counter
    reservoir.M += 1;

    // Update the weight sum
    reservoir.weightSum += risWeight;

    // Decide if we will randomly pick this sample
    bool selectSample = (random * reservoir.weightSum < risWeight);

    // If we did select this sample, update the relevant data.
    // New samples don't have visibility or age information, we can skip that.
    if (selectSample) 
    {
        reservoir.lightData = lightIndex | RESTIR_Reservoir_LightValidBit;
        reservoir.uvData = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);
        reservoir.targetPdf = targetPdf;
    }

    return selectSample;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This is a very general form, allowing input parameters to specfiy normalization and targetPdf
// rather than computing them from `newReservoir`.  Named "internal" since these parameters take
// different meanings (e.g., in RESTIR_CombineReservoirs() or RESTIR_StreamNeighborWithPairwiseMIS())
bool RESTIR_InternalSimpleResample(
    inout RESTIR_Reservoir reservoir,
    const RESTIR_Reservoir newReservoir,
    float random,
    float targetPdf RESTIR_DEFAULT(1.0f),            // Usually closely related to the sample normalization, 
    float sampleNormalization RESTIR_DEFAULT(1.0f),  //     typically off by some multiplicative factor 
    float sampleM RESTIR_DEFAULT(1.0f)               // In its most basic form, should be newReservoir.M
)
{
    // What's the current weight (times any prior-step RIS normalization factor)
    float risWeight = targetPdf * sampleNormalization;

    // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
    reservoir.M += sampleM;

    // Update the weight sum
    reservoir.weightSum += risWeight;

    // Decide if we will randomly pick this sample
    bool selectSample = (random * reservoir.weightSum < risWeight);

    // If we did select this sample, update the relevant data
    if (selectSample)
    {
        reservoir.lightData = newReservoir.lightData;
        reservoir.uvData = newReservoir.uvData;
        reservoir.targetPdf = targetPdf;
        reservoir.packedVisibility = newReservoir.packedVisibility;
        reservoir.spatialDistance = newReservoir.spatialDistance;
        reservoir.age = newReservoir.age;
    }

    return selectSample;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// Algorithm (4) from the ReSTIR paper, Combining the streams of multiple reservoirs.
// Normalization - Equation (6) - is postponed until all reservoirs are combined.
bool RESTIR_CombineReservoirs(
    inout RESTIR_Reservoir reservoir,
    const RESTIR_Reservoir newReservoir,
    float random,
    float targetPdf)
{
    return RESTIR_InternalSimpleResample(
        reservoir,
        newReservoir,
        random,
        targetPdf,
        newReservoir.weightSum * newReservoir.M,
        newReservoir.M
    );
}

// Performs normalization of the reservoir after streaming. Equation (6) from the ReSTIR paper.
void RESTIR_FinalizeResampling(
    inout RESTIR_Reservoir reservoir,
    float normalizationNumerator,
    float normalizationDenominator)
{
    float denominator = reservoir.targetPdf * normalizationDenominator;

    reservoir.weightSum = (denominator == 0.0) ? 0.0 : (reservoir.weightSum * normalizationNumerator) / denominator;
}

// A helper used for pairwise MIS computations.  This might be able to simplify code elsewhere, too.
float RESTIR_TargetPdfHelper(const RESTIR_Reservoir lightReservoir, const RAB_Surface surface, bool priorFrame RESTIR_DEFAULT(false))
{
    RAB_LightSample lightSample = RAB_SamplePolymorphicLight(
        RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(lightReservoir), priorFrame),
        surface, RESTIR_GetReservoirSampleUV(lightReservoir));

    return RAB_GetLightSampleTargetPdfForSurface(lightSample, surface);
}

// "Pairwise MIS" is a MIS approach that is O(N) instead of O(N^2) for N estimators.  The idea is you know
// a canonical sample which is a known (pretty-)good estimator, but you'd still like to improve the result
// given multiple other candidate estimators.  You can do this in a pairwise fashion, MIS'ing between each
// candidate and the canonical sample.  RESTIR_StreamNeighborWithPairwiseMIS() is executed once for each 
// candidate, after which the MIS is completed by calling RESTIR_StreamCanonicalWithPairwiseStep() once for
// the canonical sample.
// See Chapter 9.1 of https://digitalcommons.dartmouth.edu/dissertations/77/, especially Eq 9.10 & Algo 8
bool RESTIR_StreamNeighborWithPairwiseMIS(inout RESTIR_Reservoir reservoir,
    float random,
    const RESTIR_Reservoir neighborReservoir,
    const RAB_Surface neighborSurface,
    const RESTIR_Reservoir canonicalReservor,
    const RAB_Surface canonicalSurface,
    const uint numberOfNeighborsInStream)    // # neighbors streamed via pairwise MIS before streaming the canonical sample
{
    // Compute PDFs of the neighbor and cannonical light samples and surfaces in all permutations.
    // Note: First two must be computed this way.  Last two *should* be replacable by neighborReservoir.targetPdf
    // and canonicalReservor.targetPdf to reduce redundant computations, but there's a bug in that naive reuse.
    float neighborWeightAtCanonical = max(0.0f, RESTIR_TargetPdfHelper(neighborReservoir, canonicalSurface, false));
    float canonicalWeightAtNeighbor = max(0.0f, RESTIR_TargetPdfHelper(canonicalReservor, neighborSurface, false));
    float neighborWeightAtNeighbor = max(0.0f, RESTIR_TargetPdfHelper(neighborReservoir, neighborSurface, false));
    float canonicalWeightAtCanonical = max(0.0f, RESTIR_TargetPdfHelper(canonicalReservor, canonicalSurface, false));

    // Compute two pairwise MIS weights
    float w0 = RESTIR_PairwiseMisWeight(neighborWeightAtNeighbor, neighborWeightAtCanonical,
        neighborReservoir.M * numberOfNeighborsInStream, canonicalReservor.M);
    float w1 = RESTIR_PairwiseMisWeight(canonicalWeightAtNeighbor, canonicalWeightAtCanonical,
        neighborReservoir.M * numberOfNeighborsInStream, canonicalReservor.M);

    // Determine the effective M value when using pairwise MIS
    float M = neighborReservoir.M * min(
        RESTIR_MFactor(neighborWeightAtNeighbor, neighborWeightAtCanonical),
        RESTIR_MFactor(canonicalWeightAtNeighbor, canonicalWeightAtCanonical));

    // With pairwise MIS, we touch the canonical sample multiple times (but every other sample only once).  This 
    // with overweight the canonical sample; we track how much it is overweighted so we can renormalize to account
    // for this in the function RESTIR_StreamCanonicalWithPairwiseStep()
    reservoir.canonicalWeight += (1.0f - w1);

    // Go ahead and stream the neighbor sample through via RIS, appropriately weighted
    return RESTIR_InternalSimpleResample(reservoir, neighborReservoir, random,
        neighborWeightAtCanonical,
        neighborReservoir.weightSum * w0,
        M);
}

// Called to finish the process of doing pairwise MIS.  This function must be called after all required calls to
// RESTIR_StreamNeighborWithPairwiseMIS(), since pairwise MIS overweighs the canonical sample.  This function 
// compensates for this overweighting, but it can only happen after all neighbors have been processed.
bool RESTIR_StreamCanonicalWithPairwiseStep(inout RESTIR_Reservoir reservoir,
    float random,
    const RESTIR_Reservoir canonicalReservoir,
    const RAB_Surface canonicalSurface)
{
    return RESTIR_InternalSimpleResample(reservoir, canonicalReservoir, random,
        canonicalReservoir.targetPdf,
        canonicalReservoir.weightSum * reservoir.canonicalWeight,
        canonicalReservoir.M);
}

void RESTIR_SamplePdfMipmap(
    inout RAB_RandomSamplerState rng, 
    RESTIR_TEX2D pdfTexture, // full mip chain starting from unnormalized sampling pdf in mip 0
    uint2 pdfTextureSize,        // dimensions of pdfTexture at mip 0; must be 16k or less
    out uint2 position,
    out float pdf)
{
    int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))) - 1);

    position = uint2(0, 0);
    pdf = 1.0;
    for (int mipLevel = lastMipLevel; mipLevel >= 0; mipLevel--)
    {
        position *= 2;

        float4 samples;
        samples.x = max(0, RESTIR_TEX2D_LOAD(pdfTexture, int2(position.x + 0, position.y + 0), mipLevel).x);
        samples.y = max(0, RESTIR_TEX2D_LOAD(pdfTexture, int2(position.x + 0, position.y + 1), mipLevel).x);
        samples.z = max(0, RESTIR_TEX2D_LOAD(pdfTexture, int2(position.x + 1, position.y + 0), mipLevel).x);
        samples.w = max(0, RESTIR_TEX2D_LOAD(pdfTexture, int2(position.x + 1, position.y + 1), mipLevel).x);

        float weightSum = samples.x + samples.y + samples.z + samples.w;
        if (weightSum <= 0)
        {
            pdf = 0;
            return;
        }

        samples /= weightSum;

        float rnd = RAB_GetNextRandom(rng);
        
        int2 selectedOffset;

        if (rnd < samples.x)
        { 
            pdf *= samples.x;
        }
        else
        {
            rnd -= samples.x;
            
            if (rnd < samples.y)
            {
                position += uint2(0, 1);
                pdf *= samples.y;
            }
            else
            {
                rnd -= samples.y;

                if (rnd < samples.z)
                {
                    position += uint2(1, 0);
                    pdf *= samples.z;
                }
                else
                {
                    position += uint2(1, 1);
                    pdf *= samples.w;
                }
            }
        }
    }
}

#if RESTIR_ENABLE_PRESAMPLING

void RESTIR_PresampleLocalLights(
    inout RAB_RandomSamplerState rng, 
    RESTIR_TEX2D pdfTexture,
    uint2 pdfTextureSize,
    uint tileIndex,
    uint sampleInTile,
    RESTIR_ResamplingRuntimeParameters params)
{
    uint2 texelPosition;
    float pdf;
    RESTIR_SamplePdfMipmap(rng, pdfTexture, pdfTextureSize, texelPosition, pdf);

    uint lightIndex = RESTIR_ZCurveToLinearIndex(texelPosition);

    uint risBufferPtr = sampleInTile + tileIndex * params.tileSize;

    bool compact = false;
    float invSourcePdf = 0;

    if (pdf > 0)
    {
        invSourcePdf = 1.0 / pdf;

        RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex + params.firstLocalLight, false);
        compact = RAB_StoreCompactLightInfo(risBufferPtr, lightInfo);
    }

    lightIndex += params.firstLocalLight;

    if(compact) {
        lightIndex |= RESTIR_LIGHT_COMPACT_BIT;
    }

    // Store the index of the light that we found and its inverse pdf.
    // Or zero and zero if we somehow found nothing.
    RESTIR_RIS_BUFFER[risBufferPtr] = uint2(lightIndex, asuint(invSourcePdf));
}

void RESTIR_PresampleEnvironmentMap(
    inout RAB_RandomSamplerState rng, 
    RESTIR_TEX2D pdfTexture,
    uint2 pdfTextureSize,
    uint tileIndex,
    uint sampleInTile,
    RESTIR_ResamplingRuntimeParameters params)
{
    uint2 texelPosition;
    float pdf;
    RESTIR_SamplePdfMipmap(rng, pdfTexture, pdfTextureSize, texelPosition, pdf);

    // Uniform sampling inside the pixels
    float2 fPos = float2(texelPosition);
    fPos.x += RAB_GetNextRandom(rng);
    fPos.y += RAB_GetNextRandom(rng);
    
    // Convert texel position to UV and pack it
    float2 uv = fPos / float2(pdfTextureSize);
    uint packedUv = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);

    // Compute the inverse PDF if we found something
    pdf *= pdfTextureSize.x * pdfTextureSize.y;
    float invSourcePdf = (pdf > 0) ? (1.0 / pdf) : 0;

    // Store the result
    uint risBufferPtr = params.environmentRisBufferOffset + sampleInTile + tileIndex * params.environmentTileSize;
    RESTIR_RIS_BUFFER[risBufferPtr] = uint2(packedUv, asuint(invSourcePdf));
}

#endif // RESTIR_ENABLE_PRESAMPLING

#ifndef RESTIR_TILE_SIZE_IN_PIXELS
#define RESTIR_TILE_SIZE_IN_PIXELS 16
#endif

// SDK internal function that samples the given set of lights generated by RIS
// or the local light pool. The RIS set can come from local light importance presampling or from ReGIR.
RESTIR_Reservoir RESTIR_SampleLocalLightsInternal(
    inout RAB_RandomSamplerState rng, 
    RAB_Surface surface, 
    RESTIR_SampleParameters sampleParams,
    RESTIR_ResamplingRuntimeParameters params,
#if RESTIR_ENABLE_PRESAMPLING
    bool useRisBuffer,
    uint risBufferBase,
    uint risBufferCount,
#endif
    out RAB_LightSample o_selectedSample)
{
    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    o_selectedSample = RAB_EmptyLightSample();

    if (params.numLocalLights == 0)
        return state;

    if (sampleParams.numLocalLightSamples == 0)
        return state;

    for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        float rnd = RAB_GetNextRandom(rng);

        uint rndLight;
        RAB_LightInfo lightInfo = RAB_EmptyLightInfo();
        float invSourcePdf;
        bool lightLoaded = false;

#if RESTIR_ENABLE_PRESAMPLING
        if (useRisBuffer)
        {
            uint risSample = min(uint(floor(rnd * risBufferCount)), risBufferCount - 1);
            uint risBufferPtr = risSample + risBufferBase;
            
            uint2 tileData = RESTIR_RIS_BUFFER[risBufferPtr];
            rndLight = tileData.x & RESTIR_LIGHT_INDEX_MASK;
            invSourcePdf = asfloat(tileData.y);

            if ((tileData.x & RESTIR_LIGHT_COMPACT_BIT) != 0)
            {
                lightInfo = RAB_LoadCompactLightInfo(risBufferPtr);
                lightLoaded = true;
            }
        }
        else
#endif
        {
            rndLight = min(uint(floor(rnd * params.numLocalLights)), params.numLocalLights - 1) + params.firstLocalLight;
            invSourcePdf = float(params.numLocalLights);
        }

        if (!lightLoaded) {
            lightInfo = RAB_LoadLightInfo(rndLight, false);
        }

        float2 uv;
        uv.x = RAB_GetNextRandom(rng);
        uv.y = RAB_GetNextRandom(rng);

        RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
        float blendedSourcePdf = RESTIR_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
            sampleParams.localLightMisWeight, false, sampleParams);
        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = RAB_GetNextRandom(rng);

        if (blendedSourcePdf == 0)
        {
            continue;
        }
        bool selected = RESTIR_StreamSample(state, rndLight, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

        if (selected) {
            o_selectedSample = candidateSample;
        }
    }
    
    RESTIR_FinalizeResampling(state, 1.0, sampleParams.numMisSamples);
    state.M = 1;

    return state;
}

// Samples the local light pool for the given surface.
RESTIR_Reservoir RESTIR_SampleLocalLights(
    inout RAB_RandomSamplerState rng, 
    inout RAB_RandomSamplerState coherentRng,
    RAB_Surface surface, 
    RESTIR_SampleParameters sampleParams,
    RESTIR_ResamplingRuntimeParameters params,
    out RAB_LightSample o_selectedSample)
{
    float tileRnd = RAB_GetNextRandom(coherentRng);
    uint tileIndex = uint(tileRnd * params.tileCount);

    uint risBufferBase = tileIndex * params.tileSize;

    return RESTIR_SampleLocalLightsInternal(rng, surface, sampleParams, params,
#if RESTIR_ENABLE_PRESAMPLING
        params.enableLocalLightImportanceSampling != 0, risBufferBase, params.tileSize,
#endif
        o_selectedSample);
}

// Samples the infinite light pool for the given surface.
RESTIR_Reservoir RESTIR_SampleInfiniteLights(
    inout RAB_RandomSamplerState rng, 
    RAB_Surface surface, 
    uint numSamples,
    RESTIR_ResamplingRuntimeParameters params,
    out RAB_LightSample o_selectedSample)
{
    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    o_selectedSample = RAB_EmptyLightSample();

    if (params.numInfiniteLights == 0)
        return state;

    if (numSamples == 0)
        return state;

    uint stride = (params.numInfiniteLights + numSamples - 1) / numSamples;
    uint randStart = uint(RAB_GetNextRandom(rng) * params.numInfiniteLights);

    float invSourcePdf = float(params.numInfiniteLights);

    for(uint i = 0; i < numSamples; i++)
    {
        float rnd = RAB_GetNextRandom(rng);

        uint rndLight = params.firstInfiniteLight 
            + min(uint(floor(rnd * params.numInfiniteLights)), params.numInfiniteLights - 1);

        float2 uv;
        uv.x = RAB_GetNextRandom(rng);
        uv.y = RAB_GetNextRandom(rng);

        RAB_LightInfo lightInfo = RAB_LoadLightInfo(rndLight, false);
        
        RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
        
        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = RAB_GetNextRandom(rng);

        bool selected = RESTIR_StreamSample(state, rndLight, uv, risRnd, targetPdf, invSourcePdf);

        if(selected)
        {
            o_selectedSample = candidateSample;
        }
    }

    RESTIR_FinalizeResampling(state, 1.0, state.M);
    state.M = 1;

    return state;
}

#if RESTIR_ENABLE_PRESAMPLING

RESTIR_Reservoir RESTIR_SampleEnvironmentMap(
    inout RAB_RandomSamplerState rng, 
    inout RAB_RandomSamplerState coherentRng,
    RAB_Surface surface, 
    RESTIR_SampleParameters sampleParams,
    RESTIR_ResamplingRuntimeParameters params,
    out RAB_LightSample o_selectedSample)
{
    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    o_selectedSample = RAB_EmptyLightSample();

    if (params.environmentLightPresent == 0)
        return state;

    if (sampleParams.numEnvironmentMapSamples == 0)
        return state;

    float tileRnd = RAB_GetNextRandom(coherentRng);
    uint tileIndex = uint(tileRnd * params.environmentTileCount);

    uint risBufferBase = tileIndex * params.environmentTileSize + params.environmentRisBufferOffset;
    uint risBufferCount = params.environmentTileSize;

    RAB_LightInfo lightInfo = RAB_LoadLightInfo(params.environmentLightIndex, false);

    for (uint i = 0; i < sampleParams.numEnvironmentMapSamples; i++)
    {
        float rnd = RAB_GetNextRandom(rng);
        uint risSample = min(uint(floor(rnd * risBufferCount)), risBufferCount - 1);
        uint risBufferPtr = risSample + risBufferBase;
        
        uint2 tileData = RESTIR_RIS_BUFFER[risBufferPtr];
        uint packedUv = tileData.x;
        float invSourcePdf = asfloat(tileData.y);

        float2 uv = float2(packedUv & 0xffff, packedUv >> 16) / float(0xffff);        

        RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);

        float blendedSourcePdf = RESTIR_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
            sampleParams.environmentMapMisWeight, true, sampleParams);
        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = RAB_GetNextRandom(rng);

        bool selected = RESTIR_StreamSample(state, params.environmentLightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

        if (selected) {
            o_selectedSample = candidateSample;
        }
    }

    RESTIR_FinalizeResampling(state, 1.0, sampleParams.numMisSamples);
    state.M = 1;

    return state;
}

#endif // RESTIR_ENABLE_PRESAMPLING

// Samples from the BRDF defined by the given surface
RESTIR_Reservoir RESTIR_SampleBrdf(
    inout RAB_RandomSamplerState rng,
    RAB_Surface surface,
    RESTIR_SampleParameters sampleParams,
    RESTIR_ResamplingRuntimeParameters params,
    out RAB_LightSample o_selectedSample)
{
    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    
    for (uint i = 0; i < sampleParams.numBrdfSamples; ++i)
    {
        float lightSourcePdf = 0;
        float3 sampleDir;
        uint lightIndex = RESTIR_InvalidLightIndex;
        float2 randXY = float2(0, 0);
        RAB_LightSample candidateSample = RAB_EmptyLightSample();

        if (RAB_GetSurfaceBrdfSample(surface, rng, sampleDir))
        {
            float brdfPdf = RAB_GetSurfaceBrdfPdf(surface, sampleDir);
            float maxDistance = RESTIR_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
            
            bool hitAnything = RAB_TraceRayForLocalLight(RAB_GetSurfaceWorldPos(surface), sampleDir,
                sampleParams.brdfRayMinT, maxDistance, lightIndex, randXY);

            if (lightIndex != RESTIR_InvalidLightIndex)
            {
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
                candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
                    
                if (sampleParams.brdfCutoff > 0.f)
                {
                    // If Mis cutoff is used, we need to evaluate the sample and make sure it actually could have been
                    // generated by the area sampling technique. This is due to numerical precision.
                    float3 lightDir;
                    float lightDistance;
                    RAB_GetLightDirDistance(surface, candidateSample, lightDir, lightDistance);

                    float brdfPdf = RAB_GetSurfaceBrdfPdf(surface, lightDir);
                    float maxDistance = RESTIR_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
                    if (lightDistance > maxDistance)
                        lightIndex = RESTIR_InvalidLightIndex;
                }

                if (lightIndex != RESTIR_InvalidLightIndex)
                {
                    lightSourcePdf = RAB_EvaluateLocalLightSourcePdf(params, lightIndex);
                }
            }
            else if (!hitAnything && params.environmentLightPresent != 0)
            {
                // sample environment light
                lightIndex = params.environmentLightIndex;
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
                randXY = RAB_GetEnvironmentMapRandXYFromDir(sampleDir);
                candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
                lightSourcePdf = RAB_EvaluateEnvironmentMapSamplingPdf(sampleDir);
            }
        }

        if (lightSourcePdf == 0)
        {
            // Did not hit a visible light
            continue;
        }

        bool isEnvMapSample = lightIndex == params.environmentLightIndex;
        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float blendedSourcePdf = RESTIR_LightBrdfMisWeight(surface, candidateSample, lightSourcePdf,
            isEnvMapSample ? sampleParams.environmentMapMisWeight : sampleParams.localLightMisWeight, 
            isEnvMapSample,
            sampleParams);
        float risRnd = RAB_GetNextRandom(rng);

        bool selected = RESTIR_StreamSample(state, lightIndex, randXY, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected) {
            o_selectedSample = candidateSample;
        }
    }

    RESTIR_FinalizeResampling(state, 1.0, sampleParams.numMisSamples);
    state.M = 1;

    return state;
}

// Samples ReGIR and the local and infinite light pools for a given surface.
RESTIR_Reservoir RESTIR_SampleLightsForSurface(
    inout RAB_RandomSamplerState rng,
    inout RAB_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RESTIR_SampleParameters sampleParams,
    RESTIR_ResamplingRuntimeParameters params, 
    out RAB_LightSample o_lightSample)
{
    o_lightSample = RAB_EmptyLightSample();

    RESTIR_Reservoir localReservoir;
    RAB_LightSample localSample = RAB_EmptyLightSample();

    localReservoir = RESTIR_SampleLocalLights(rng, coherentRng, surface, 
        sampleParams, params, localSample);

    RAB_LightSample infiniteSample = RAB_EmptyLightSample();  
    RESTIR_Reservoir infiniteReservoir = RESTIR_SampleInfiniteLights(rng, surface,
        sampleParams.numInfiniteLightSamples, params, infiniteSample);

#if RESTIR_ENABLE_PRESAMPLING
    RAB_LightSample environmentSample = RAB_EmptyLightSample();
    RESTIR_Reservoir environmentReservoir = RESTIR_SampleEnvironmentMap(rng, coherentRng, surface,
        sampleParams, params, environmentSample);
#endif

    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RESTIR_Reservoir brdfReservoir = RESTIR_SampleBrdf(rng, surface, sampleParams, params, brdfSample);

    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    RESTIR_CombineReservoirs(state, localReservoir, 0.5, localReservoir.targetPdf);
    bool selectInfinite = RESTIR_CombineReservoirs(state, infiniteReservoir, RAB_GetNextRandom(rng), infiniteReservoir.targetPdf);
#if RESTIR_ENABLE_PRESAMPLING
    bool selectEnvironment = RESTIR_CombineReservoirs(state, environmentReservoir, RAB_GetNextRandom(rng), environmentReservoir.targetPdf);
#endif
    bool selectBrdf = RESTIR_CombineReservoirs(state, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);
    
    RESTIR_FinalizeResampling(state, 1.0, 1.0);
    state.M = 1;

    if (selectBrdf)
        o_lightSample = brdfSample;
    else
#if RESTIR_ENABLE_PRESAMPLING
    if (selectEnvironment)
        o_lightSample = environmentSample;
    else
#endif
    if (selectInfinite)
        o_lightSample = infiniteSample;
    else
        o_lightSample = localSample;

    return state;
}

#ifdef RESTIR_ENABLE_BOILING_FILTER
// RESTIR_BOILING_FILTER_GROUP_SIZE must be defined - 16 is a reasonable value
#define RESTIR_BOILING_FILTER_MIN_LANE_COUNT 32

groupshared float s_weights[(RESTIR_BOILING_FILTER_GROUP_SIZE * RESTIR_BOILING_FILTER_GROUP_SIZE + RESTIR_BOILING_FILTER_MIN_LANE_COUNT - 1) / RESTIR_BOILING_FILTER_MIN_LANE_COUNT];
groupshared uint s_count[(RESTIR_BOILING_FILTER_GROUP_SIZE * RESTIR_BOILING_FILTER_GROUP_SIZE + RESTIR_BOILING_FILTER_MIN_LANE_COUNT - 1) / RESTIR_BOILING_FILTER_MIN_LANE_COUNT];

// Boiling filter that should be applied at the end of the temporal resampling pass.
// Can be used inside the same shader that does temporal resampling if it's a compute shader,
// or in a separate pass if temporal resampling is a raygen shader.
// The filter analyzes the weights of all reservoirs in a thread group, and discards
// the reservoirs whose weights are very high, i.e. above a certain threshold.
void RESTIR_BoilingFilter(
    uint2 LocalIndex,
    float filterStrength, // (0..1]
    RESTIR_ResamplingRuntimeParameters params,
    inout RESTIR_Reservoir state)
{
    // Boiling happens when some highly unlikely light is discovered and it is relevant
    // for a large surface area around the pixel that discovered it. Then this light sample
    // starts to propagate to the neighborhood through spatiotemporal reuse, which looks like
    // a flash. We can detect such lights because their weight is significantly higher than 
    // the weight of their neighbors. So, compute the average group weight and apply a threshold.

    float boilingFilterMultiplier = 10.f / clamp(filterStrength, 1e-6, 1.0) - 9.f;

    // Start with average nonzero weight within the wavefront
    float waveWeight = WaveActiveSum(state.weightSum);
    uint waveCount = WaveActiveCountBits(state.weightSum > 0);

    // Store the results of each wavefront into shared memory
    uint linearThreadIndex = LocalIndex.x + LocalIndex.y * RESTIR_BOILING_FILTER_GROUP_SIZE;
    uint waveIndex = linearThreadIndex / WaveGetLaneCount();

    if (WaveIsFirstLane())
    {
        s_weights[waveIndex] = waveWeight;
        s_count[waveIndex] = waveCount;
    }

    GroupMemoryBarrierWithGroupSync();

    // Reduce the per-wavefront averages into a global average using one wavefront
    if (linearThreadIndex < (RESTIR_BOILING_FILTER_GROUP_SIZE * RESTIR_BOILING_FILTER_GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount())
    {
        waveWeight = s_weights[linearThreadIndex];
        waveCount = s_count[linearThreadIndex];

        waveWeight = WaveActiveSum(waveWeight);
        waveCount = WaveActiveSum(waveCount);

        if (linearThreadIndex == 0)
        {
            s_weights[0] = (waveCount > 0) ? (waveWeight / float(waveCount)) : 0.0;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Read the per-group average and apply the threshold
    float averageNonzeroWeight = s_weights[0];
    if (state.weightSum > averageNonzeroWeight * boilingFilterMultiplier)
    {
        state = RESTIR_EmptyReservoir();
    }
}
#endif // RESTIR_ENABLE_BOILING_FILTER


// Internal SDK function that permutes the pixels sampled from the previous frame.
void RESTIR_ApplyPermutationSampling(inout int2 prevPixelPos, uint uniformRandomNumber)
{
    int2 offset = int2(uniformRandomNumber & 3, (uniformRandomNumber >> 2) & 3);
    prevPixelPos += offset;
 
    prevPixelPos.x ^= 3;
    prevPixelPos.y ^= 3;
    
    prevPixelPos -= offset;
}

// A structure that groups the application-provided settings for temporal resampling.
struct RESTIR_TemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;
};

// Temporal resampling pass.
// Takes the previous G-buffer, motion vectors, and two light reservoir buffers as inputs.
// Tries to match the surfaces in the current frame to surfaces in the previous frame.
// If a match is found for a given pixel, the current and previous reservoirs are 
// combined. An optional visibility ray may be cast if enabled, to reduce the resampling bias.
// That visibility ray should ideally be traced through the previous frame BVH, but
// can also use the current frame BVH if the previous is not available - that will produce more bias.
// The selectedLightSample parameter is used to update and return the selected sample; it's optional,
// and it's safe to pass a null structure there and ignore the result.
RESTIR_Reservoir RESTIR_TemporalResampling(
    uint2 pixelPosition,
    RAB_Surface surface,
    RESTIR_Reservoir curSample,
    RAB_RandomSamplerState rng,
    RESTIR_TemporalResamplingParameters tparams,
    RESTIR_ResamplingRuntimeParameters params,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    // For temporal reuse, there's only a pair of samples; pairwise and basic MIS are essentially identical
    if (tparams.biasCorrectionMode == RESTIR_BIAS_CORRECTION_PAIRWISE)
    {
        tparams.biasCorrectionMode = RESTIR_BIAS_CORRECTION_BASIC;
    }

    uint historyLimit = min(RESTIR_PackedReservoir_MaxM, uint(tparams.maxHistoryLength * curSample.M));

    int selectedLightPrevID = -1;

    if (RESTIR_IsValidReservoir(curSample))
    {
        selectedLightPrevID = RAB_TranslateLightIndex(RESTIR_GetReservoirLightIndex(curSample), true);
    }

    temporalSamplePixelPos = int2(-1, -1);

    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    RESTIR_CombineReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

    // Backproject this pixel to last frame
    float3 motion = tparams.screenSpaceMotion;
    
    if (!tparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }

    float2 reprojectedSamplePosition = float2(pixelPosition) + motion.xy;
    int2 prevPos = int2(round(reprojectedSamplePosition));

    float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

    RAB_Surface temporalSurface = RAB_EmptySurface();
    bool foundNeighbor = false;
    const float radius = (params.activeCheckerboardField == 0) ? 4 : 8;
    int2 spatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    for(int i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if(i > 0)
        {
            offset.x = int((RAB_GetNextRandom(rng) - 0.5) * radius);
            offset.y = int((RAB_GetNextRandom(rng) - 0.5) * radius);
        }

        int2 idx = prevPos + offset;
        if (tparams.enablePermutationSampling && i == 0)
        {
            RESTIR_ApplyPermutationSampling(idx, params.uniformRandomNumber);
        }

        if (!RESTIR_IsActiveCheckerboardPixel(idx, true, params))
        {
            idx.x += int(params.activeCheckerboardField) * 2 - 3;
        }

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;
        
        // Test surface similarity, discard the sample if the surface is too different.
        if (!RESTIR_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface), 
            tparams.normalThreshold, tparams.depthThreshold))
            continue;

        spatialOffset = idx - prevPos;
        prevPos = idx;
        foundNeighbor = true;

        break;
    }

    bool selectedPreviousSample = false;
    float previousM = 0;

    if (foundNeighbor)
    {
        // Resample the previous frame sample into the current reservoir, but reduce the light's weight
        // according to the bilinear weight of the current pixel
        uint2 prevReservoirPos = RESTIR_PixelPosToReservoir(prevPos, params);
        RESTIR_Reservoir prevSample = RESTIR_LoadReservoir(params,
            prevReservoirPos, tparams.sourceBufferIndex);
        prevSample.M = min(prevSample.M, historyLimit);
        prevSample.spatialDistance += spatialOffset;
        prevSample.age += 1;

        uint originalPrevLightID = RESTIR_GetReservoirLightIndex(prevSample);

        // Map the light ID from the previous frame into the current frame, if it still exists
        if (RESTIR_IsValidReservoir(prevSample))
        {
            if (prevSample.age <= 1)
            {
                temporalSamplePixelPos = prevPos;
            }

            int mappedLightID = RAB_TranslateLightIndex(RESTIR_GetReservoirLightIndex(prevSample), false);

            if (mappedLightID < 0)
            {
                // Kill the reservoir
                prevSample.weightSum = 0;
                prevSample.lightData = 0;
            }
            else
            {
                // Sample is valid - modify the light ID stored
                prevSample.lightData = mappedLightID | RESTIR_Reservoir_LightValidBit;
            }
        }

        previousM = prevSample.M;

        float weightAtCurrent = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RESTIR_IsValidReservoir(prevSample))
        {
            const RAB_LightInfo candidateLight = RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(prevSample), false);

            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, surface, RESTIR_GetReservoirSampleUV(prevSample));

            weightAtCurrent = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        bool sampleSelected = RESTIR_CombineReservoirs(state, prevSample, RAB_GetNextRandom(rng), weightAtCurrent);
        if(sampleSelected)
        {
            selectedPreviousSample = true;
            selectedLightPrevID = int(originalPrevLightID);
            selectedLightSample = candidateLightSample;
        }
    }

#if RESTIR_ALLOWED_BIAS_CORRECTION >= RESTIR_BIAS_CORRECTION_BASIC
    if (tparams.biasCorrectionMode >= RESTIR_BIAS_CORRECTION_BASIC)
    {
        // Compute the unbiased normalization term (instead of using 1/M)
        float pi = state.targetPdf;
        float piSum = state.targetPdf * curSample.M;
        
        if (RESTIR_IsValidReservoir(state) && selectedLightPrevID >= 0 && previousM > 0)
        {
            float temporalP = 0;

            const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

            // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
            const RAB_LightSample selectedSampleAtTemporal = RAB_SamplePolymorphicLight(
                selectedLightPrev, temporalSurface, RESTIR_GetReservoirSampleUV(state));
        
            temporalP = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtTemporal, temporalSurface);

#if RESTIR_ALLOWED_BIAS_CORRECTION >= RESTIR_BIAS_CORRECTION_RAY_TRACED
            if (tparams.biasCorrectionMode == RESTIR_BIAS_CORRECTION_RAY_TRACED && temporalP > 0 && (!selectedPreviousSample || !tparams.enableVisibilityShortcut))
            {
                if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, selectedSampleAtTemporal))
                {
                    temporalP = 0;
                }
            }
#endif

            pi = selectedPreviousSample ? temporalP : pi;
            piSum += temporalP * previousM;
        }

        RESTIR_FinalizeResampling(state, pi, piSum);
    }
    else
#endif
    {
        RESTIR_FinalizeResampling(state, 1.0, state.M);
    }

    return state;
}

// A structure that groups the application-provided settings for spatial resampling.
struct RESTIR_SpatialResamplingParameters
{
    // The index of the reservoir buffer to pull the spatial samples from.
    uint sourceBufferIndex;
    
    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint numSamples;

    // Number of neighbor pixels considered when there is not enough history data (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    uint numDisocclusionBoostSamples;

    // Disocclusion boost is activated when the current reservoir's M value
    // is less than targetHistoryLength.
    uint targetHistoryLength;


    // Controls the bias correction math for spatial reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray *per every spatial sample* per pixel 
    // (or per two pixels if checkerboard sampling is enabled).
    uint biasCorrectionMode;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Surface depth similarity threshold for spatial reuse.
    // See 'RESTIR_TemporalResamplingParameters::depthThreshold' for more information.
    float depthThreshold;

    // Surface normal similarity threshold for spatial reuse.
    // See 'RESTIR_TemporalResamplingParameters::normalThreshold' for more information.
    float normalThreshold;
};

// Spatial resampling pass, using pairwise MIS.  
// Inputs and outputs equivalent to RESTIR_SpatialResampling(), but only uses pairwise MIS.
// Can call this directly, or call RESTIR_SpatialResampling() with sparams.biasCorrectionMode 
// set to RESTIR_BIAS_CORRECTION_PAIRWISE, which simply calls this function.
RESTIR_Reservoir RESTIR_SpatialResamplingWithPairwiseMIS(
    uint2 pixelPosition,
    RAB_Surface centerSurface,
    RESTIR_Reservoir centerSample,
    RAB_RandomSamplerState rng,
    RESTIR_SpatialResamplingParameters sparams,
    RESTIR_ResamplingRuntimeParameters params,
    inout RAB_LightSample selectedLightSample)
{
    // Initialize the output reservoir
    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    state.canonicalWeight = 0.0f;

    // How many spatial samples to use?  
    uint numSpatialSamples = (centerSample.M < sparams.targetHistoryLength)
        ? max(sparams.numDisocclusionBoostSamples, sparams.numSamples)
        : sparams.numSamples;

    // Walk the specified number of neighbors, resampling using RIS
    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);
    uint validSpatialSamples = 0;
    uint i;
    for (i = 0; i < numSpatialSamples; ++i)
    {
        // Get screen-space location of neighbor
        uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
        int2 spatialOffset = int2(float2(RESTIR_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius);
        int2 idx = int2(pixelPosition)+spatialOffset;

        if (!RESTIR_IsActiveCheckerboardPixel(idx, false, params))
            idx.x += (idx.y & 1) != 0 ? 1 : -1;

        RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, false);

        // Check for surface / G-buffer matches between the canonical sample and this neighbor
        if (!RAB_IsSurfaceValid(neighborSurface))
            continue;

        if (!RESTIR_IsValidNeighbor(RAB_GetSurfaceNormal(centerSurface), RAB_GetSurfaceNormal(neighborSurface),
            RAB_GetSurfaceLinearDepth(centerSurface), RAB_GetSurfaceLinearDepth(neighborSurface),
            sparams.normalThreshold, sparams.depthThreshold))
            continue;

        if (!RAB_AreMaterialsSimilar(centerSurface, neighborSurface))
            continue;

        // The surfaces are similar enough so we *can* reuse a neighbor from this pixel, so load it.
        RESTIR_Reservoir neighborSample = RESTIR_LoadReservoir(params,
            RESTIR_PixelPosToReservoir(idx, params), sparams.sourceBufferIndex);
        neighborSample.spatialDistance += spatialOffset;

        validSpatialSamples++;

        // If sample has weight 0 due to visibility (or etc), skip the expensive-ish MIS computations
        if (neighborSample.M <= 0) continue;

        // Stream this light through the reservoir using pairwise MIS
        RESTIR_StreamNeighborWithPairwiseMIS(state, RAB_GetNextRandom(rng),
            neighborSample, neighborSurface,   // The spatial neighbor
            centerSample, centerSurface,       // The canonical (center) sample
            numSpatialSamples);
    }

    // If we've seen no usable neighbor samples, set the weight of the central one to 1
    state.canonicalWeight = (validSpatialSamples <= 0) ? 1.0f : state.canonicalWeight;

    // Stream the canonical sample (i.e., from prior computations at this pixel in this frame) using pairwise MIS.
    RESTIR_StreamCanonicalWithPairwiseStep(state, RAB_GetNextRandom(rng), centerSample, centerSurface);

    RESTIR_FinalizeResampling(state, 1.0, float(max(1, validSpatialSamples)));

    // Return the selected light sample.  This is a redundant lookup and could be optimized away by storing
        // the selected sample from the stream steps above.
    selectedLightSample = RAB_SamplePolymorphicLight(
        RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(state), false),
        centerSurface, RESTIR_GetReservoirSampleUV(state));

    return state;
}


// Spatial resampling pass.
// Operates on the current frame G-buffer and its reservoirs.
// For each pixel, considers a number of its neighbors and, if their surfaces are 
// similar enough to the current pixel, combines their light reservoirs.
// Optionally, one visibility ray is traced for each neighbor being considered, to reduce bias.
// The selectedLightSample parameter is used to update and return the selected sample; it's optional,
// and it's safe to pass a null structure there and ignore the result.
RESTIR_Reservoir RESTIR_SpatialResampling(
    uint2 pixelPosition,
    RAB_Surface centerSurface,
    RESTIR_Reservoir centerSample,
    RAB_RandomSamplerState rng,
    RESTIR_SpatialResamplingParameters sparams,
    RESTIR_ResamplingRuntimeParameters params,
    inout RAB_LightSample selectedLightSample)
{
    if (sparams.biasCorrectionMode == RESTIR_BIAS_CORRECTION_PAIRWISE)
    {
        return RESTIR_SpatialResamplingWithPairwiseMIS(pixelPosition, centerSurface, 
            centerSample, rng, sparams, params, selectedLightSample);
    }

    RESTIR_Reservoir state = RESTIR_EmptyReservoir();

    // This is the weight we'll use (instead of 1/M) to make our estimate unbaised (see paper).
    float normalizationWeight = 1.0f;

    // Since we're using our bias correction scheme, we need to remember which light selection we made
    int selected = -1;

    RAB_LightInfo selectedLight = RAB_EmptyLightInfo();

    if (RESTIR_IsValidReservoir(centerSample))
    {
        selectedLight = RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(centerSample), false);
    }

    RESTIR_CombineReservoirs(state, centerSample, /* random = */ 0.5f, centerSample.targetPdf);

    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);
    
    uint i;
    uint numSpatialSamples = sparams.numSamples;
    if(centerSample.M < sparams.targetHistoryLength)
        numSpatialSamples = max(sparams.numDisocclusionBoostSamples, numSpatialSamples);

    // Clamp the sample count at 32 to make sure we can keep the neighbor mask in an uint (cachedResult)
    numSpatialSamples = min(numSpatialSamples, 32);

    // We loop through neighbors twice.  Cache the validity / edge-stopping function
    //   results for the 2nd time through.
    uint cachedResult = 0;

    // Walk the specified number of neighbors, resampling using RIS
    for (i = 0; i < numSpatialSamples; ++i)
    {
        // Get screen-space location of neighbor
        uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
        int2 spatialOffset = int2(float2(RESTIR_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius);
        int2 idx = int2(pixelPosition) + spatialOffset;

        if (!RESTIR_IsActiveCheckerboardPixel(idx, false, params))
            idx.x += (idx.y & 1) != 0 ? 1 : -1;

        RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, false);

        if (!RAB_IsSurfaceValid(neighborSurface))
            continue;

        if (!RESTIR_IsValidNeighbor(RAB_GetSurfaceNormal(centerSurface), RAB_GetSurfaceNormal(neighborSurface), 
            RAB_GetSurfaceLinearDepth(centerSurface), RAB_GetSurfaceLinearDepth(neighborSurface), 
            sparams.normalThreshold, sparams.depthThreshold))
            continue;

        if (!RAB_AreMaterialsSimilar(centerSurface, neighborSurface))
            continue;

        uint2 neighborReservoirPos = RESTIR_PixelPosToReservoir(idx, params);

        RESTIR_Reservoir neighborSample = RESTIR_LoadReservoir(params,
            neighborReservoirPos, sparams.sourceBufferIndex);
        neighborSample.spatialDistance += spatialOffset;

        cachedResult |= (1u << uint(i));

        RAB_LightInfo candidateLight = RAB_EmptyLightInfo();

        // Load that neighbor's RIS state, do resampling
        float neighborWeight = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RESTIR_IsValidReservoir(neighborSample))
        {   
            candidateLight = RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(neighborSample), false);
            
            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, centerSurface, RESTIR_GetReservoirSampleUV(neighborSample));
            
            neighborWeight = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, centerSurface);
        }
        
        if (RESTIR_CombineReservoirs(state, neighborSample, RAB_GetNextRandom(rng), neighborWeight))
        {
            selected = int(i);
            selectedLight = candidateLight;
            selectedLightSample = candidateLightSample;
        }
    }

    if (RESTIR_IsValidReservoir(state))
    {
#if RESTIR_ALLOWED_BIAS_CORRECTION >= RESTIR_BIAS_CORRECTION_BASIC
        if (sparams.biasCorrectionMode >= RESTIR_BIAS_CORRECTION_BASIC)
        {
            // Compute the unbiased normalization term (instead of using 1/M)
            float pi = state.targetPdf;
            float piSum = state.targetPdf * centerSample.M;

            // To do this, we need to walk our neighbors again
            for (i = 0; i < numSpatialSamples; ++i)
            {
                // If we skipped this neighbor above, do so again.
                if ((cachedResult & (1u << uint(i))) == 0) continue;

                uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;

                // Get the screen-space location of our neighbor
                int2 idx = int2(pixelPosition) + int2(float2(RESTIR_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius);

                if (!RESTIR_IsActiveCheckerboardPixel(idx, false, params))
                    idx.x += (idx.y & 1) != 0 ? 1 : -1;

                // Load our neighbor's G-buffer
                RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, false);
                
                // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
                const RAB_LightSample selectedSampleAtNeighbor = RAB_SamplePolymorphicLight(
                    selectedLight, neighborSurface, RESTIR_GetReservoirSampleUV(state));

                float ps = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, neighborSurface);

#if RESTIR_ALLOWED_BIAS_CORRECTION >= RESTIR_BIAS_CORRECTION_RAY_TRACED
                if (sparams.biasCorrectionMode == RESTIR_BIAS_CORRECTION_RAY_TRACED && ps > 0)
                {
                    if (!RAB_GetConservativeVisibility(neighborSurface, selectedSampleAtNeighbor))
                    {
                        ps = 0;
                    }
                }
#endif

                uint2 neighborReservoirPos = RESTIR_PixelPosToReservoir(idx, params);

                RESTIR_Reservoir neighborSample = RESTIR_LoadReservoir(params,
                    neighborReservoirPos, sparams.sourceBufferIndex);

                // Select this sample for the (normalization) numerator if this particular neighbor pixel
                //     was the one we selected via RIS in the first loop, above.
                pi = selected == i ? ps : pi;

                // Add to the sums of weights for the (normalization) denominator
                piSum += ps * neighborSample.M;
            }

            // Use "MIS-like" normalization
            RESTIR_FinalizeResampling(state, pi, piSum);
        }
        else
#endif
        {
            RESTIR_FinalizeResampling(state, 1.0, state.M);
        }
    }

    return state;
}


// A structure that groups the application-provided settings for spatio-temporal resampling.
struct RESTIR_SpatioTemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint numSamples;

    // Number of neighbor pixels considered when there is no temporal surface (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    uint numDisocclusionBoostSamples;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;
};

// Fused spatialtemporal resampling pass, using pairwise MIS.  
// Inputs and outputs equivalent to RESTIR_SpatioTemporalResampling(), but only uses pairwise MIS.
// Can call this directly, or call RESTIR_SpatioTemporalResampling() with sparams.biasCorrectionMode 
// set to RESTIR_BIAS_CORRECTION_PAIRWISE, which simply calls this function.
RESTIR_Reservoir RESTIR_SpatioTemporalResamplingWithPairwiseMIS(
    uint2 pixelPosition,
    RAB_Surface surface,
    RESTIR_Reservoir curSample,
    RAB_RandomSamplerState rng,
    RESTIR_SpatioTemporalResamplingParameters stparams,
    RESTIR_ResamplingRuntimeParameters params,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    uint historyLimit = min(RESTIR_PackedReservoir_MaxM, uint(stparams.maxHistoryLength * curSample.M));

    // Backproject this pixel to last frame
    float3 motion = stparams.screenSpaceMotion;
    if (!stparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }
    int2 prevPos = int2(round(float2(pixelPosition)+motion.xy));
    float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

    // Some default initializations
    temporalSamplePixelPos = int2(-1, -1);
    RAB_Surface temporalSurface = RAB_EmptySurface();
    bool foundTemporalSurface = false;                                                 // Found a valid backprojection?
    const float temporalSearchRadius = (params.activeCheckerboardField == 0) ? 4 : 8;  // How far to search for a match when backprojecting
    int2 temporalSpatialOffset = int2(0, 0);                                           // Offset for the (central) backprojected pixel

    // Try to find a matching surface in the neighborhood of the centrol reprojected pixel
    int i;
    int2 centralIdx;
    for (i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        offset.x = (i > 0) ? int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius) : 0;
        offset.y = (i > 0) ? int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius) : 0;

        centralIdx = prevPos + offset;
        if (stparams.enablePermutationSampling && i == 0)
        {
            RESTIR_ApplyPermutationSampling(centralIdx, params.uniformRandomNumber);
        }

        if (!RESTIR_IsActiveCheckerboardPixel(centralIdx, true, params))
        {
            centralIdx.x += int(params.activeCheckerboardField) * 2 - 3;
        }

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(centralIdx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;

        // Test surface similarity, discard the sample if the surface is too different.
        if (!RESTIR_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface),
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface),
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        temporalSpatialOffset = centralIdx - prevPos;
        foundTemporalSurface = true;
        break;
    }

    // How many spatial samples to use?  
    uint numSpatialSamples = (!foundTemporalSurface)
        ? max(stparams.numDisocclusionBoostSamples, stparams.numSamples)
        : uint(int(stparams.numSamples));

    // Count how many of our spatiotemporal samples are valid and streamed via RIS
    int validSamples = 0;

    // Create an empty reservoir we'll use to accumulate into
    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    state.canonicalWeight = 0.0f;    // Important this is 0 for temporal

    // Load the "temporal" reservoir at the temporally backprojected "central" pixel
    RESTIR_Reservoir prevSample = RESTIR_LoadReservoir(params,
        RESTIR_PixelPosToReservoir(centralIdx, params), stparams.sourceBufferIndex);
    prevSample.M = min(prevSample.M, historyLimit);
    prevSample.spatialDistance += temporalSpatialOffset;
    prevSample.age += 1;

    // Find the prior frame's light in the current frame
    int mappedLightID = RAB_TranslateLightIndex(RESTIR_GetReservoirLightIndex(prevSample), false);

    // Kill the reservoir if it doesn't exist in the current frame, otherwise update its ID for this frame
    prevSample.weightSum = (mappedLightID < 0) ? 0 : prevSample.weightSum;
    prevSample.lightData = (mappedLightID < 0) ? 0 : mappedLightID | RESTIR_Reservoir_LightValidBit;

    // If we found a valid surface by backprojecting our current pixel, stream it through the reservoir.
    if (foundTemporalSurface && prevSample.M > 0)
    {
        ++validSamples;

        // Pass out the temporal sample location
        temporalSamplePixelPos = (prevSample.age <= 1) ? centralIdx : temporalSamplePixelPos;

        // Stream this light through the reservoir using pairwise MIS
        RESTIR_StreamNeighborWithPairwiseMIS(state, RAB_GetNextRandom(rng),
            prevSample, temporalSurface,    // The temporal neighbor
            curSample, surface,             // The canonical neighbor
            1 + numSpatialSamples);
    }

    // Look for valid (spatiotemporal) neighbors and stream them through the reservoir via pairwise MIS
    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);
    for (i = 1; i < numSpatialSamples; ++i)
    {
        uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
        int2 spatialOffset = int2(float2(RESTIR_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);
        int2 idx = prevPos + spatialOffset;

        if (idx.x < 0 || idx.y < 0)
            continue;

        if (!RESTIR_IsActiveCheckerboardPixel(idx, false, params))
        {
            idx.x += (idx.y & 1) != 0 ? 1 : -1;
        }

        RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, true);

        // Check for surface / G-buffer matches between the canonical sample and this neighbor
        if (!RAB_IsSurfaceValid(neighborSurface))
            continue;

        if (!RESTIR_IsValidNeighbor(RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(neighborSurface),
            RAB_GetSurfaceLinearDepth(surface), RAB_GetSurfaceLinearDepth(neighborSurface),
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        if (!RAB_AreMaterialsSimilar(surface, neighborSurface))
            continue;

        // The surfaces are similar enough so we *can* reuse a neighbor from this pixel, so load it.
        RESTIR_Reservoir neighborSample = RESTIR_LoadReservoir(params,
            RESTIR_PixelPosToReservoir(idx, params), stparams.sourceBufferIndex);
        neighborSample.M = min(neighborSample.M, historyLimit);
        neighborSample.spatialDistance += spatialOffset;
        neighborSample.age += 1;

        // Find the this neighbors light in the current frame (it may have turned off or moved in the ID list)
        int mappedLightID = RAB_TranslateLightIndex(RESTIR_GetReservoirLightIndex(neighborSample), false);

        // Kill the sample if the light doesn't exist in the current frame, otherwise update its ID for this frame
        neighborSample.weightSum = (mappedLightID < 0) ? 0 : neighborSample.weightSum;
        neighborSample.lightData = (mappedLightID < 0) ? 0 : mappedLightID | RESTIR_Reservoir_LightValidBit;

        if (mappedLightID < 0) continue;

        ++validSamples;

        // If sample has weight 0 due to visibility (or etc), skip the expensive-ish MIS computations
        if (neighborSample.M <= 0) continue;

        // Stream this light through the reservoir using pairwise MIS
        RESTIR_StreamNeighborWithPairwiseMIS(state, RAB_GetNextRandom(rng),
            neighborSample, neighborSurface,   // The spatial neighbor
            curSample, surface,                // The canonical (center) sample
            1 + numSpatialSamples);
    }

    // Stream the canonical sample (i.e., from prior computations at this pixel in this frame) using pairwise MIS.
    RESTIR_StreamCanonicalWithPairwiseStep(state, RAB_GetNextRandom(rng),
        curSample, surface);

    // Renormalize the reservoir so it can be stored in a packed format 
    RESTIR_FinalizeResampling(state, 1.0f, float(max(1, validSamples)));

    // Return the selected light sample.  This is a redundant lookup and could be optimized away by storing
    // the selected sample from the stream steps above.
    selectedLightSample = RAB_SamplePolymorphicLight(
        RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(state), false),
        surface, RESTIR_GetReservoirSampleUV(state));

    return state;
}


// Spatio-temporal resampling pass.
// A combination of the temporal and spatial passes that operates only on the previous frame reservoirs.
// The selectedLightSample parameter is used to update and return the selected sample; it's optional,
// and it's safe to pass a null structure there and ignore the result.
RESTIR_Reservoir RESTIR_SpatioTemporalResampling(
    uint2 pixelPosition,
    RAB_Surface surface,
    RESTIR_Reservoir curSample,
    RAB_RandomSamplerState rng,
    RESTIR_SpatioTemporalResamplingParameters stparams,
    RESTIR_ResamplingRuntimeParameters params,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    if (stparams.biasCorrectionMode == RESTIR_BIAS_CORRECTION_PAIRWISE)
    {
        return RESTIR_SpatioTemporalResamplingWithPairwiseMIS(pixelPosition, surface,
            curSample, rng, stparams, params, temporalSamplePixelPos, selectedLightSample);
    }

    uint historyLimit = min(RESTIR_PackedReservoir_MaxM, uint(stparams.maxHistoryLength * curSample.M));

    int selectedLightPrevID = -1;

    if (RESTIR_IsValidReservoir(curSample))
    {
        selectedLightPrevID = RAB_TranslateLightIndex(RESTIR_GetReservoirLightIndex(curSample), true);
    }

    temporalSamplePixelPos = int2(-1, -1);

    RESTIR_Reservoir state = RESTIR_EmptyReservoir();
    RESTIR_CombineReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);

    // Backproject this pixel to last frame
    float3 motion = stparams.screenSpaceMotion;

    if (!stparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }

    float2 reprojectedSamplePosition = float2(pixelPosition) + motion.xy;
    int2 prevPos = int2(round(reprojectedSamplePosition));

    float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

    int i;

    RAB_Surface temporalSurface = RAB_EmptySurface();
    bool foundTemporalSurface = false;
    const float temporalSearchRadius = (params.activeCheckerboardField == 0) ? 4 : 8;
    int2 temporalSpatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    for (i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if (i > 0)
        {
            offset.x = int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius);
            offset.y = int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius);
        }

        int2 idx = prevPos + offset;

        if (stparams.enablePermutationSampling && i == 0)
        {
            RESTIR_ApplyPermutationSampling(idx, params.uniformRandomNumber);
        }

        if (!RESTIR_IsActiveCheckerboardPixel(idx, true, params))
        {
            idx.x += int(params.activeCheckerboardField) * 2 - 3;
        }

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;
        
        // Test surface similarity, discard the sample if the surface is too different.
        if (!RESTIR_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface), 
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        temporalSpatialOffset = idx - prevPos;
        foundTemporalSurface = true;
        break;
    }

    // Clamp the sample count at 32 to make sure we can keep the neighbor mask in an uint (cachedResult)
    uint numSamples = clamp(stparams.numSamples, 1, 32);

    // Apply disocclusion boost if there is no temporal surface
    if (!foundTemporalSurface)
        numSamples = clamp(stparams.numDisocclusionBoostSamples, numSamples, 32);

    // We loop through neighbors twice.  Cache the validity / edge-stopping function
    //   results for the 2nd time through.
    uint cachedResult = 0;

    // Since we're using our bias correction scheme, we need to remember which light selection we made
    int selected = -1;

    // Walk the specified number of neighbors, resampling using RIS
    for (i = 0; i < numSamples; ++i)
    {
        int2 spatialOffset, idx;

        // Get screen-space location of neighbor
        if (i == 0 && foundTemporalSurface)
        {
            spatialOffset = temporalSpatialOffset;
            idx = prevPos + spatialOffset;
        }
        else
        {
            uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
            spatialOffset = (i == 0 && foundTemporalSurface) 
                ? temporalSpatialOffset 
                : int2(float2(RESTIR_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);

            idx = prevPos + spatialOffset;

            if (!RESTIR_IsActiveCheckerboardPixel(idx, true, params))
            {
                idx.x += int(params.activeCheckerboardField) * 2 - 3;
            }

            temporalSurface = RAB_GetGBufferSurface(idx, true);

            if (!RAB_IsSurfaceValid(temporalSurface))
                continue;

            if (!RESTIR_IsValidNeighbor(RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
                RAB_GetSurfaceLinearDepth(surface), RAB_GetSurfaceLinearDepth(temporalSurface), 
                stparams.normalThreshold, stparams.depthThreshold))
                continue;

            if (!RAB_AreMaterialsSimilar(surface, temporalSurface))
                continue;
        }
        
        cachedResult |= (1u << uint(i));

        uint2 neighborReservoirPos = RESTIR_PixelPosToReservoir(idx, params);

        RESTIR_Reservoir prevSample = RESTIR_LoadReservoir(params,
            neighborReservoirPos, stparams.sourceBufferIndex);

        prevSample.M = min(prevSample.M, historyLimit);
        prevSample.spatialDistance += spatialOffset;
        prevSample.age += 1;

        uint originalPrevLightID = RESTIR_GetReservoirLightIndex(prevSample);

        // Map the light ID from the previous frame into the current frame, if it still exists
        if (RESTIR_IsValidReservoir(prevSample))
        {   
            if (i == 0 && foundTemporalSurface && prevSample.age <= 1)
            {
                temporalSamplePixelPos = idx;
            }

            int mappedLightID = RAB_TranslateLightIndex(RESTIR_GetReservoirLightIndex(prevSample), false);

            if (mappedLightID < 0)
            {
                // Kill the reservoir
                prevSample.weightSum = 0;
                prevSample.lightData = 0;
            }
            else
            {
                // Sample is valid - modify the light ID stored
                prevSample.lightData = mappedLightID | RESTIR_Reservoir_LightValidBit;
            }
        }

        RAB_LightInfo candidateLight;

        // Load that neighbor's RIS state, do resampling
        float neighborWeight = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RESTIR_IsValidReservoir(prevSample))
        {   
            candidateLight = RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(prevSample), false);
            
            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, surface, RESTIR_GetReservoirSampleUV(prevSample));
            
            neighborWeight = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        if (RESTIR_CombineReservoirs(state, prevSample, RAB_GetNextRandom(rng), neighborWeight))
        {
            selected = i;
            selectedLightPrevID = int(originalPrevLightID);
            selectedLightSample = candidateLightSample;
        }
    }

    if (RESTIR_IsValidReservoir(state))
    {
#if RESTIR_ALLOWED_BIAS_CORRECTION >= RESTIR_BIAS_CORRECTION_BASIC
        if (stparams.biasCorrectionMode >= RESTIR_BIAS_CORRECTION_BASIC)
        {
            // Compute the unbiased normalization term (instead of using 1/M)
            float pi = state.targetPdf;
            float piSum = state.targetPdf * curSample.M;

            if (selectedLightPrevID >= 0)
            {
                const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

                // To do this, we need to walk our neighbors again
                for (i = 0; i < numSamples; ++i)
                {
                    // If we skipped this neighbor above, do so again.
                    if ((cachedResult & (1u << uint(i))) == 0) continue;

                    uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;

                    // Get the screen-space location of our neighbor
                    int2 spatialOffset = (i == 0 && foundTemporalSurface) 
                        ? temporalSpatialOffset 
                        : int2(float2(RESTIR_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);
                    int2 idx = prevPos + spatialOffset;

                    if (!RESTIR_IsActiveCheckerboardPixel(idx, true, params))
                    {
                        idx.x += int(params.activeCheckerboardField) * 2 - 3;
                    }

                    // Load our neighbor's G-buffer
                    RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, true);
                    
                    // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
                    const RAB_LightSample selectedSampleAtNeighbor = RAB_SamplePolymorphicLight(
                        selectedLightPrev, neighborSurface, RESTIR_GetReservoirSampleUV(state));

                    float ps = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, neighborSurface);

#if RESTIR_ALLOWED_BIAS_CORRECTION >= RESTIR_BIAS_CORRECTION_RAY_TRACED
                                                                                                              // TODO:  WHY?
                    if (stparams.biasCorrectionMode == RESTIR_BIAS_CORRECTION_RAY_TRACED && ps > 0 && (selected != i || i != 0 || !stparams.enableVisibilityShortcut))
                    {
                        RAB_Surface fallbackSurface;
                        if (i == 0 && foundTemporalSurface)
                            fallbackSurface = surface;
                        else
                            fallbackSurface = neighborSurface;

                        if (!RAB_GetTemporalConservativeVisibility(fallbackSurface, neighborSurface, selectedSampleAtNeighbor))
                        {
                            ps = 0;
                        }
                    }
#endif

                    uint2 neighborReservoirPos = RESTIR_PixelPosToReservoir(idx, params);

                    RESTIR_Reservoir prevSample = RESTIR_LoadReservoir(params,
                        neighborReservoirPos, stparams.sourceBufferIndex);
                    prevSample.M = min(prevSample.M, historyLimit);

                    // Select this sample for the (normalization) numerator if this particular neighbor pixel
                    //     was the one we selected via RIS in the first loop, above.
                    pi = selected == i ? ps : pi;

                    // Add to the sums of weights for the (normalization) denominator
                    piSum += ps * prevSample.M;
                }
            }

            // Use "MIS-like" normalization
            RESTIR_FinalizeResampling(state, pi, piSum);
        }
        else
#endif
        {
            RESTIR_FinalizeResampling(state, 1.0, state.M);
        }
    }

    return state;
}

#endif // RESAMPLING_FUNCTIONS_HLSLI
