/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import PackedTypes;
import Scene.RaytracingInline;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
import SurfaceData;

// Thread groups size for launches. Not user controllable, due to internal RESTIRGDI restrictions
static const uint kRESTIRGDIGroupSize = 16;

// Only sample specular lobe when BRDF sampling is enabled.
static const bool kSampleSpecularOnly = false;

/** Global data structure used to hold parameters and resources for RESTIRGDI.
*/
struct ReSTIR
{
    RESTIR_ResamplingRuntimeParameters params;           ///< RESTIRGDI runtime parameters (potentially changing every frame).

    uint    frameIndex;                                 ///< Current frame index. Used to make sure we use different random numbers each frame.
    uint2   frameDim;                                   ///< Frame dimensions in pixels.
    uint    pixelCount;                                 ///< Equivalent to frameDim.x * frameDim.y.
    float   rayEpsilon;                                 ///< Our ray-epsilon.
    uint    currentSurfaceBufferIndex;                  ///< Index of the surface data buffer for the current frame (0 or 1)
    uint    prevSurfaceBufferIndex;                     ///< Index of the surface data buffer for the previous frame (0 or 1).

    // Camera data for the previous frame.
    float3  prevCameraU;
    float3  prevCameraV;
    float3  prevCameraW;
    float2  prevCameraJitter;

    bool    storeCompactLightInfo;                      ///< Enable storing compact light info for presampled light tiles.
    bool    useEmissiveTextures;                        ///< Lookup final triangle emission in emissive texture.

    // Parameters to control how RESTIRGDI generates initial light candidate samples.
    uint    localLightCandidateCount;                   ///< Number of initial candidates (per pixel) on local lights.
    uint    infiniteLightCandidateCount;                ///< Number of initial candidates (per pixel) on infinite lights.
    uint    envLightCandidateCount;                     ///< Number of initial candidates (per pixel) on the environment light.
    uint    brdfCandidateCount;                         ///< Number of initial candidates (per pixel) from brdf.
    float   brdfCutoff;                                 ///< Value in range[0, 1] to determine how much to shorten BRDF rays. 0 to disable shortening.

    // Parameters needed to control RESTIRGDI sample reuse (generally).
    uint    maxHistoryLength;                           ///< Maximum history length for temporal reuse, measured in frames.
    uint    biasCorrectionMode;                         ///< See definition of BiasCorrection in RESTIRGDI.h.
    float   depthThreshold;                             ///< Relative depth difference at which pixels are classified too far apart to be reused (0.1 = 10%).
    float   normalThreshold;                            ///< Cosine of the angle between normals, below which pixels are classified too far apart to be reused.

    // Parameters needed to control RESTIRGDI's spatial reuse (and spatiotemporal reuse).
    float   samplingRadius;                             ///< Screen-space radius for spatial resampling, measured in pixels.
    uint    spatialSampleCount;                         ///< Number of neighbor pixels considered for resampling.
    uint    disocclusionSampleCount;                    ///< Number of neighbor pixels considered for resampling in disoccluded regions.

    // Parameters needed to control RESTIRGDI's temporal reuse (and spatiotemporal reuse).
    float   boilingFilterStrength;                      ///< 0 = off, 1 = full strength.
    bool    enableVisibilityShortcut;                   ///< Should we use a visibility shortcut?  (Degrades quality somewhat)
    bool    enablePermutationSampling;                  ///< Enables permuting the pixels sampled from the previous frame (noisier but more denoiser friendly).

    // Parameters only valid during shading (i.e., after calling RESTIRGDI::update() in your C++ code).
    uint    finalShadingReservoir;                      ///< Reservoir where the final samples are stored (needed to extract them for the user).

    // Texture and buffer resources used in the RESTIRGDI bridge.
    StructuredBuffer<PackedPolymorphicLight> lightInfo;             ///< Buffer storing information about all the lights in the scene.
    RWBuffer<uint2> risBuffer;                                      ///< Buffer storing presampled light tiles.
    RWStructuredBuffer<PackedPolymorphicLight> compactLightInfo;    ///< Optional buffer storing compact light info for samples in the light tile buffer for improved coherence.
    RWStructuredBuffer<PackedSurfaceData> surfaceData;              ///< Buffer storing the surface data for the current and previous frames.
    Buffer<float2> neighborOffsets;                                 ///< Buffer storing a poisson(-ish) distribution of offsets for sampling randomized neighbors.
    RWStructuredBuffer<RESTIR_PackedReservoir> reservoirs;           ///< Buffer storing reservoirs.
    Texture2D<float2> motionVectors;                                ///< Texture storing screen-space motion vectors.

    Texture2D localLightPdfTexture;                     ///< PDF texture for sampling local lights (emissive triangles, local analytical lights).
    Texture2D envLightLuminanceTexture;                 ///< Luminance texture for evaluating the environment light.
    Texture2D envLightPdfTexture;                       ///< PDF texture for sampling environment light.
};

// HACK: PolymorphicLight.slang contains light sampling functions accessing data in gRESTIRGDI.
// We also use PolymorphicLight.slang in the LightUpdater.cs.slang module for setting up
// light information. The global gRESTIRGDI resources are not available there, so we disable
// the light sampling functions by default.
// Here, we enable the light sampling functions and #include the module rather than
// importing it, so all the code is inlined into this module.
#define RESTIRGDI_POLYMORPHIC_LIGHT_SAMPLING
#include "PolymorphicLight.slang"

// ------------------------------------------------------------------------------------------------
// Types.
// ------------------------------------------------------------------------------------------------

/** Stores information about a surface, including its position, orientation, and material parameters.
    See SurfaceData.slang for details.
*/
typedef SurfaceData RAB_Surface;

/** Stores information about a polymorphic light, i.e. a light of any type.
    See PolymorphicLight.slang for details.
*/
typedef PolymorphicLight RAB_LightInfo;

/** Represents a point on a light and its radiance, weighted relative to the surface that was used
    to generate the sample.
    See PolymorphicLight.slang for details.
*/
typedef PolymorphicLightSample RAB_LightSample;

/** Stores the mutable state of a random number generator (RNG).
    We use Falcor's TinyUniformSampleGenerator which is based on a 32-bit LCG.
    This PRNG seems good enough and using a PCG-based replacement doesn't give any perceptible
    benefit (but increases cost).
*/
typedef TinyUniformSampleGenerator RAB_RandomSamplerState;

// ------------------------------------------------------------------------------------------------
// Empty type constructors.
// ------------------------------------------------------------------------------------------------

RAB_Surface RAB_EmptySurface()
{
    return {};
}

RAB_LightInfo RAB_EmptyLightInfo()
{
    return {};
}

RAB_LightSample RAB_EmptyLightSample()
{
    return {};
}

// ------------------------------------------------------------------------------------------------
// G-buffer input and accessor functions.
// ------------------------------------------------------------------------------------------------

/** A copy of Falcor's Camera::computeRayPinhole() method that works for current/last frame.
 */
float3 computeRayDirection(uint2 pixel, bool previousFrame)
{
    const float2 jitter = previousFrame
        ? float2(-gRESTIRGDI.prevCameraJitter.x, gRESTIRGDI.prevCameraJitter.y)
                                        : float2(-gScene.camera.data.jitterX, gScene.camera.data.jitterY);

    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    const float2 p = (pixel + float2(0.5f, 0.5f)) / gRESTIRGDI.frameDim + jitter;
    const float2 ndc = float2(2, -2) * p + float2(-1, 1);

    const float3 cameraU = previousFrame ? gRESTIRGDI.prevCameraU : gScene.camera.data.cameraU;
    const float3 cameraV = previousFrame ? gRESTIRGDI.prevCameraV : gScene.camera.data.cameraV;
    const float3 cameraW = previousFrame ? gRESTIRGDI.prevCameraW : gScene.camera.data.cameraW;

    // Compute the normalized ray direction assuming a pinhole camera.
    return normalize(ndc.x * cameraU + ndc.y * cameraV + cameraW);
}

/** Loads a surface from the current or previous G-buffer at the specified pixel position.
    Pixel positions may be out-of-bounds or negative, in which case the function is supposed to
    return an invalid surface.
*/
RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    // Return invalid surface data if pixel is out of bounds.
    if (any(pixelPosition < 0) || any(pixelPosition >= gRESTIRGDI.frameDim)) return SurfaceData();

    // Compute buffer index based on pixel and currently used surface buffer index.
    uint surfaceIdx = previousFrame ? gRESTIRGDI.prevSurfaceBufferIndex : gRESTIRGDI.currentSurfaceBufferIndex;
    uint bufferIdx = pixelPosition.x + (pixelPosition.y * gRESTIRGDI.frameDim.x) + surfaceIdx * gRESTIRGDI.pixelCount;

    // Compute ray direction.
    float3 rayDir = computeRayDirection(pixelPosition, previousFrame);

    // Load surface data.
    return SurfaceData(gRESTIRGDI.surfaceData[bufferIdx], -rayDir);
}

/** Tests if the provided surface contains valid geometry.
 */
bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return surface.valid;
}

/** Returns the world position of the provided surface.
 */
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    return surface.position;
}

/** Returns the world space shading normal of the provided surface.
 */
float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return surface.normal;
}

/** Returns the linear depth of the provided surface.
 */
float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return max(0.f, surface.viewDepth);
}

// ------------------------------------------------------------------------------------------------
// Lights and samples.
// ------------------------------------------------------------------------------------------------

/** Loads the information about a polymorphic light based on its index, on the current
    or previous frame.
    Note: Our current implementation does only maintain one copy of light information, so the
    previousFrame argument is ignored.
*/
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return PolymorphicLight(gRESTIRGDI.lightInfo[index]);
}

/** Stores the information about a polymorphic light in a compacted form in the RIS buffer location
    at linearIndex.
*/
bool RAB_StoreCompactLightInfo(uint linearIndex, RAB_LightInfo lightInfo)
{
    if (gRESTIRGDI.storeCompactLightInfo) return false;

    gRESTIRGDI.compactLightInfo[linearIndex] = lightInfo.pack();
    return true;
}

/** Loads the information about a polymorphic light that is stored in a compacted form in the
    RIS buffer location at linearIndex.
*/
RAB_LightInfo RAB_LoadCompactLightInfo(uint linearIndex)
{
    return PolymorphicLight(gRESTIRGDI.compactLightInfo[linearIndex]);
}

/** Translates the light index from the current frame to the previous frame
    (if currentToPrevious is true) or from the previous frame to the current frame
    (if currentToPrevious is false).
    Returns the new index, or a negative number if the light does not exist in the other frame.
    Note: Our current implementation does only maintain one copy of light information,
    so we just return the same index.
*/
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    return lightIndex;
}

/** Returns the direction and distance from the surface to the light sample.
 */
void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir,
                             out float o_lightDistance)
{
    o_lightDir = lightSample.direction;
    o_lightDistance = lightSample.distance;
}

/** Returns true if the light sample comes from an analytic light
    (e.g. a sphere or rectangle primitive) that cannot be sampled by BRDF rays.
*/
bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return lightSample.lightType == PolymorphicLightType::Analytic;
}

/** Returns the solid angle PDF of the light sample.
 */
float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.pdf;
}

/** Computes the probability of a particular direction being sampled from the environment map
    relative to all the other possible directions, based on the environment map PDF texture.
*/
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
    float2 uv = gScene.envMap.worldToUv(L);

    uint2 pdfTextureSize;
    gRESTIRGDI.envLightPdfTexture.GetDimensions(pdfTextureSize.x, pdfTextureSize.y);
    uint2 texelPosition = uint2(pdfTextureSize * uv);
    float texelValue = gRESTIRGDI.envLightPdfTexture.Load(uint3(texelPosition, 0)).x;

    int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))) - 1);
    float averageValue = 0.5f * (
        gRESTIRGDI.envLightPdfTexture.Load(uint3(0, 0, lastMipLevel)).x +
                                 gRESTIRGDI.envLightPdfTexture.Load(uint3(1, 0, lastMipLevel)).x);

    // the selection probability is multiplied by numTexels in RESTIRGDI during presampling
    // so actually numTexels cancels out in this case
    //
    // uint numTexels = pdfTextureSize.x * pdfTextureSize.y;
    // float totalSum = averageWeight * numTexels;
    // return numTexels * texelValue / totalSum;
    return texelValue / averageValue;
}

/** Computes the probability of a particular light being sampled from the local light pool
    with importance sampling, based on the local light PDF texture.
*/
float RAB_EvaluateLocalLightSourcePdf(RESTIR_ResamplingRuntimeParameters params, uint lightIndex)
{
    uint2 pdfTextureSize;
    gRESTIRGDI.localLightPdfTexture.GetDimensions(pdfTextureSize.x, pdfTextureSize.y);

    uint2 texelPosition = RESTIR_LinearIndexToZCurve(lightIndex);
    float texelValue = gRESTIRGDI.localLightPdfTexture.Load(uint3(texelPosition, 0)).r;

    int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))) - 1);
    float averageValue = 0.5f * (
        gRESTIRGDI.localLightPdfTexture.Load(uint3(0, 0, lastMipLevel)).x +
                                 gRESTIRGDI.localLightPdfTexture.Load(uint3(1, 0, lastMipLevel)).x);

    // all light pdf weight
    float sum = averageValue * pdfTextureSize.x * pdfTextureSize.y;

    return texelValue / sum;
}

// ------------------------------------------------------------------------------------------------
// Sampling functions.
// ------------------------------------------------------------------------------------------------

/** Computes the weight of the given light sample when the given surface is shaded using that
    light sample.

    This evaluates the target function (aka p-hat in the ReSTIR paper).
    We choose p-hat to evaluate the full BRDF response, minus the visibility term.
    Since it's a PDF, it's a scalar value rather than a RGB color computed when shading.
*/
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (!lightSample.valid) return 0.f;

    float3 N = surface.normal;
    float3 V = surface.viewDir;
    float3 L = lightSample.direction;
    float3 H = normalize(V + L);
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));

    // Compute the BRDF terms for the target function.
    float roughness = surface.roughness;
    float specular = surface.specular;
    float D = evalNdfGGX(roughness * roughness, NdotH);
    float G = evalMaskingSmithGGXSeparable(roughness, NdotL, NdotV);

    // This is the "right" value to match the integral
    // float F = specular < 1e-8f ? 0.f : evalFresnelSchlick(specular, 1.f, LdotH);

    // This gives much less noisy results. We can change the target PDF however we want, so use this.
    float F = specular < 1e-8f ? 0.f : (evalFresnelSchlick(specular, 1.f, LdotH) / specular);

    // Mix the diffuse and specular BRDFs.
    float diffuseBRDF = NdotL * M_1_PI;
    float specularBRDF = D * G * F / (4.f * NdotV);
    float fullBRDF = max(0.f, lerp(specularBRDF, diffuseBRDF, surface.diffuseProb));

    return fullBRDF * lightSample.Li.x;
}

/** Computes the weight of the given light for arbitrary surfaces located inside the
    specified volume. Used for world-space light grid construction (ReGIR).
    Note: Our current implementation does not support ReGIR, so this always returns zero.
*/
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter, float volumeRadius)
{
    return 0.f;
}

float3 worldToTangent(RAB_Surface surface, float3 w)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    float3 bitangent = perp_stark(surface.normal);
    float3 tangent = cross(bitangent, surface.normal);

    return float3(dot(bitangent, w), dot(tangent, w), dot(surface.normal, w));
}

float3 tangentToWorld(RAB_Surface surface, float3 h)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    float3 bitangent = perp_stark(surface.normal);
    float3 tangent = cross(bitangent, surface.normal);

    return bitangent * h.x + tangent * h.y + surface.normal * h.z;
}

/** Performs importance sampling of the surface's BRDF and returns the sampled direction.
 */
bool RAB_GetSurfaceBrdfSample(RAB_Surface surface, inout RAB_RandomSamplerState rng, out float3 dir)
{
    dir = {};

    float3 rand;
    rand.x = RAB_GetNextRandom(rng);
    rand.y = RAB_GetNextRandom(rng);
    rand.z = RAB_GetNextRandom(rng);
    if (rand.x < surface.diffuseProb)
    {
        if (kSampleSpecularOnly)
            return false;

        // consistent eval order
        float pdf;
        float3 h = sample_cosine_hemisphere_concentric(rand.yz, pdf);
        dir = tangentToWorld(surface, h);
    }
    else
    {
        float pdf;
        float ggxAlpha = surface.roughness * surface.roughness;
        float3 h = sampleGGX_NDF(ggxAlpha, rand.yz, pdf);
        dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
    }

    return dot(surface.normal, dir) > 0.f;
}

/** Computes the PDF of a particular direction being sampled by RAB_GetSurfaceBrdfSample.
 */
float RAB_GetSurfaceBrdfPdf(RAB_Surface surface, float3 dir)
{
    float cosTheta = saturate(dot(surface.normal, dir));
    float diffusePdf = kSampleSpecularOnly ? 0.f : (cosTheta / M_PI);

    float3 h = worldToTangent(surface, normalize(dir + surface.viewDir));
    float ggxAlpha = surface.roughness * surface.roughness;
    float specularPdf = evalPdfGGX_NDF(ggxAlpha, h.z) / (4.f * saturate(dot(h, worldToTangent(surface, surface.viewDir))));
    float pdf = cosTheta > 0.f ? lerp(specularPdf, diffusePdf, surface.diffuseProb) : 0.f;
    return pdf;
}

/** Samples a polymorphic light relative to the given receiver surface.
 */
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    // Calculate light sample with scalar emission.
    return lightInfo.calcSample(surface.position, uv, true);
}

/** Traces a visibility ray that returns approximate, conservative visibility between the surface
    and the light sample. Conservative means if unsure, assume the light is visible.
*/
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    Ray ray = Ray(surface.position, lightSample.direction, 0.f, lightSample.distance - gRESTIRGDI.rayEpsilon);

    // Trace our ray (with SceneRayQuery<0> meaning no alpha testing) and cull non-opaque geometry.
    SceneRayQuery<0> rayQuery;
    return rayQuery.traceVisibilityRay(ray, RAY_FLAG_CULL_NON_OPAQUE);
}

/** Same visibility ray tracing as RAB_GetConservativeVisibility but for surfaces and light samples
    originating from the previous frame.
    Note: Our implementation does not provide the last frame's BVH so we just use the current one.
    This can cause slight darkening around boundaries or fast moving objects but is generally imperceptible.
*/
bool RAB_GetTemporalConservativeVisibility(RAB_Surface curSurface, RAB_Surface prevSurface, RAB_LightSample lightSample)
{
    return RAB_GetConservativeVisibility(prevSurface, lightSample);
}

/** Converts a world-space direction into a pair of numbers that, when passed into RAB_SamplePolymorphicLight
    for the environment light, will make a sample at the same direction.
*/
float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    return gScene.envMap.worldToUv(worldDir);
}

/** Traces a ray with the given parameters, looking for a light. If a local light is found, returns true
    and fills the output parameters with the light sample information.
    \param[out] o_lightIndex If hit, must be a valid light index for RAB_LoadLightInfo, otherwise RESTIRGDI_InvalidLightIndex.
    \param[out] o_randXY The randXY that corresponds to the hit location and is the same used for RAB_SamplePolymorphicLight.
*/
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_lightIndex = RESTIR_InvalidLightIndex;
    o_randXY = 0;

    float hitT;
    SceneRayQuery<0> rayQuery;

    Ray ray = Ray(origin, direction, tMin, tMax);
    HitInfo hit = rayQuery.traceRay(ray, hitT, RAY_FLAG_CULL_NON_OPAQUE);
    if (gRESTIRGDI.params.numLocalLights && hit.isValid())
    {
        const bool isTriangleHit = hit.getType() == HitType::Triangle;

        if (isTriangleHit)
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            uint triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            float2 barycentrics = triangleHit.barycentrics;
            if (triangleIndex != LightCollection::kInvalidIndex)
            {
                uint activeLightIndex = gScene.lightCollection.getActiveTriangleIndex(triangleIndex);
                if (activeLightIndex != LightCollection::kInvalidIndex)
                {
                    // hitUVs are mapped directly to randXY in Falcor see EmissiveLight::calcSample.
                    o_randXY = barycentrics;
                    o_lightIndex = activeLightIndex;
                }
            }
        }
    }

    return hit.isValid();
}

// ------------------------------------------------------------------------------------------------
// Misc functions.
// ------------------------------------------------------------------------------------------------

/** Returns the next random number from the provided RNG state.
 */
float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return sampleNext1D(rng);
}

/** Compares the materials of two surfaces, returns true if the surfaces are similar enough that
    it makes sense to share the light reservoirs between them. A conservative implementation would
    always return true, which might result in more noisy images than actually comparing the materials.
*/
bool RAB_AreMaterialsSimilar(RAB_Surface a, RAB_Surface b)
{
    return true;
}

// ------------------------------------------------------------------------------------------------
// Helper functions not defined by RESTIRGDI.
// ------------------------------------------------------------------------------------------------

/** Initialize the random sampler.
    \param[in] index Index on 2D domain (i.e. pixel index).
    \param[in] passIndex Pass index within current frame.
    \return The random sampler.
*/
RAB_RandomSamplerState RAB_InitRandomSampler(uint2 index, uint passIndex)
{
    return TinyUniformSampleGenerator(index, gRESTIRGDI.frameIndex + passIndex * 13);
}

// The following defines are needed by ResamplingFunctions.hlsli.

// Define buffers for RESTIRGDI to access data from.
#define RESTIR_RIS_BUFFER gRESTIRGDI.risBuffer
#define RESTIR_LIGHT_RESERVOIR_BUFFER gRESTIRGDI.reservoirs
#define RESTIR_NEIGHBOR_OFFSETS_BUFFER gRESTIRGDI.neighborOffsets

// Settings needed to use the boiling filter.
#define RESTIR_ENABLE_BOILING_FILTER
#define RESTIR_BOILING_FILTER_GROUP_SIZE kRESTIRGDIGroupSize

#include "ResamplingFunctions.slang"

/** Public interface to the RESTIRGDI module.
    Note: All of these functions are no-ops if the RESTIRGDI SDK is not installed.
*/
extension ReSTIR
{
    /** Set the per-pixel surface data required for RESTIRGDI at valid surface hits.
        Note: The provided position should be save to trace visibility rays from (i.e. is offseted from the true surface).
        \param[in] pixel Pixel coordinate.
        \param[in] position Position in world-space.
        \param[in] normal Shading normal in world-space.
        \param[in] diffuse Diffuse albedo.
        \param[in] specular Specular albedo.
        \param[in] specularRoughness Specular roughness (linear).
    */
    void setSurfaceData(const uint2 pixel, const float3 position, const float3 normal, const float3 diffuse, const float3 specular, const float specularRoughness)
    {
        if (any(pixel >= gRESTIRGDI.frameDim)) return;

        // Compute buffer index based on pixel and currently used surface buffer index.
        uint bufferIdx = pixel.x + (pixel.y * gRESTIRGDI.frameDim.x) + gRESTIRGDI.currentSurfaceBufferIndex * gRESTIRGDI.pixelCount;

        SurfaceData sd = {};
        sd.position = position;
        sd.normal = normal;
        sd.viewDepth = distance(gScene.camera.getPosition(), position);

        float diffuseWeight = luminance(diffuse);
        float specularWeight = luminance(specular);
        float sumWeights = diffuseWeight + specularWeight;
        float diffuseProb = sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);

        sd.diffuse = diffuseWeight;
        sd.specular = specularWeight;
        sd.diffuseProb = diffuseProb;
        sd.roughness = specularRoughness;

        // Store surface data.
        gRESTIRGDI.surfaceData[bufferIdx] = sd.pack();
    }

    /** Mark pixel as containing no valid surface data.
       \param[in] pixel Pixel coordinate.
    */
    void setInvalidSurfaceData(uint2 pixel)
    {
        if (any(pixel >= gRESTIRGDI.frameDim)) return;

        // Compute buffer index based on pixel and currently used surface buffer index.
        uint bufferIdx = pixel.x + (pixel.y * gRESTIRGDI.frameDim.x) + gRESTIRGDI.currentSurfaceBufferIndex * gRESTIRGDI.pixelCount;

        // Store invalid surface data.
        gRESTIRGDI.surfaceData[bufferIdx] = SurfaceData().pack();
    }

    /** Get the final sample for the given pixel computed using RESTIRGDI.
        Note: The returned incident radiance is pre-divided by the PDF.
        \param[in] pixel Pixel coordinate.
        \param[out] dir Direction to light in world-space.
        \param[out] distance Distance to light.
        \param[out] Li Weighted unoccluded incident radiance (i.e., incident radiance / pdf)
        \return Returns true if a valid sample was returned.
    */
    bool getFinalSample(uint2 pixel, out float3 dir, out float distance, out float3 Li)
    {
        // Get the final sample from RESTIRGDI.
        RAB_Surface surface;
        RAB_LightSample lightSample;
        bool isValid = gRESTIRGDI.getFinalSample(pixel, surface, lightSample, Li);

        // Return data.
        dir = lightSample.direction;
        distance = lightSample.distance - gRESTIRGDI.rayEpsilon;
        return isValid;
    }
}

/** "Private" interface to the RESTIRGDI module.
    Note: The entire private API is not defined if RESTIRGDI is not installed, as they all
    use RESTIRGDI structures during computation and/or function input or output parameters.
*/
extension ReSTIR
{
    /** Generate initial light sample candidates, select one to store in the reservoir.
    */
    void generateCandidates(uint2 pixelID, uint outputReservoirIndex)
    {
        uint2 pixelPosition = RESTIR_ReservoirToPixelPos(pixelID, gRESTIRGDI.params);
        printSetPixel(pixelPosition);

        // Initialize random samplers.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 2);
        RAB_RandomSamplerState tileRng = RAB_InitRandomSampler(pixelPosition / RESTIR_TILE_SIZE_IN_PIXELS, 3);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RESTIR_SampleParameters sampleParams = RESTIR_InitSampleParameters(
            0, // Number of samples to take from ReGIR.
            gRESTIRGDI.localLightCandidateCount,
            gRESTIRGDI.infiniteLightCandidateCount,
            gRESTIRGDI.envLightCandidateCount,
            gRESTIRGDI.brdfCandidateCount,
            gRESTIRGDI.brdfCutoff,
            0.001f);

        // Generate initial candidates.
        RAB_LightSample lightSample;
        RESTIR_Reservoir reservoir = RESTIR_SampleLightsForSurface(
            rng, tileRng, surface, sampleParams, gRESTIRGDI.params, lightSample);

        // Store selected light sample in a reservoir for later reuse and shading.
        RESTIR_StoreReservoir(reservoir, gRESTIRGDI.params, pixelID, outputReservoirIndex);
    }

    /** Trace a visibility ray from the primary hit surface to the light sample in the reservoir.
        Update the reservoir if the light is not visible.
    */
    void testCandidateVisibility(uint2 pixelID, uint reservoirIndex)
    {
        // Load surface data and reservoir containing the light sample.
        RAB_Surface surface = RAB_GetGBufferSurface(RESTIR_ReservoirToPixelPos(pixelID, gRESTIRGDI.params), false);
        RESTIR_Reservoir reservoir = RESTIR_LoadReservoir(gRESTIRGDI.params, pixelID, reservoirIndex);

        if (RAB_IsSurfaceValid(surface) && RESTIR_IsValidReservoir(reservoir))
        {
            // Get the light sample, so we have data to construct our visibility query
            RAB_LightInfo lightInfo = RAB_LoadLightInfo(RESTIR_GetReservoirLightIndex(reservoir), false);
            RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, RESTIR_GetReservoirSampleUV(reservoir));

            // Trace a visibility ray and update the reservoir.
            if (!RAB_GetConservativeVisibility(surface, lightSample))
            {
                RESTIR_StoreVisibilityInReservoir(reservoir, 0, true);
                RESTIR_StoreReservoir(reservoir, gRESTIRGDI.params, pixelID, reservoirIndex);
            }
        }
    }

    /** Presample local lights into presampled light tiles.
    */
    void presampleLocalLights(uint2 threadId)
    {
        uint2 texDim;
        gRESTIRGDI.localLightPdfTexture.GetDimensions(texDim.x, texDim.y);

        RAB_RandomSamplerState rng = RAB_InitRandomSampler(threadId.xy, 0);

        RESTIR_PresampleLocalLights(
            rng,
            gRESTIRGDI.localLightPdfTexture,
            texDim,
            threadId.y,         // tileIndex [0...tileCount)
            threadId.x,         // sampleInTile [0...tileSize)
            gRESTIRGDI.params);
    }

    /** Presample environment light into presampled light tiles.
    */
    void presampleEnvLight(uint2 threadId)
    {
        uint2 texDim;
        gRESTIRGDI.envLightPdfTexture.GetDimensions(texDim.x, texDim.y);

        RAB_RandomSamplerState rng = RAB_InitRandomSampler(threadId.xy, 1);

        RESTIR_PresampleEnvironmentMap(
            rng,
            gRESTIRGDI.envLightPdfTexture,
            texDim,
            threadId.y,         // tileIndex [0...tileCount)
            threadId.x,         // sampleInTile [0...tileSize)
            gRESTIRGDI.params);
    }

    /** Perform spatial resampling.
    */
    void doSpatialResampling(uint2 pixelID, uint inReservoirIndex, uint outReservoirIndex)
    {
        uint2 pixelPosition = RESTIR_ReservoirToPixelPos(pixelID, gRESTIRGDI.params);

        // Initialize random sampler.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 4);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RESTIR_Reservoir reservoir = {};

        if (RAB_IsSurfaceValid(surface))
        {
            // Load reservoir at the current pixel.
            RESTIR_Reservoir centerReservoir = RESTIR_LoadReservoir(gRESTIRGDI.params, pixelID, inReservoirIndex);

            // Setup resampling parameters.
            RESTIR_SpatialResamplingParameters sparams;
            sparams.sourceBufferIndex = inReservoirIndex;
            sparams.numSamples = gRESTIRGDI.spatialSampleCount;
            sparams.numDisocclusionBoostSamples = gRESTIRGDI.disocclusionSampleCount;
            sparams.targetHistoryLength = gRESTIRGDI.maxHistoryLength;
            sparams.biasCorrectionMode = gRESTIRGDI.biasCorrectionMode;
            sparams.samplingRadius = gRESTIRGDI.samplingRadius;
            sparams.depthThreshold = gRESTIRGDI.depthThreshold;
            sparams.normalThreshold = gRESTIRGDI.normalThreshold;

            // Execute resampling.
            RAB_LightSample selectedSample = {};
            reservoir = RESTIR_SpatialResampling(pixelPosition, surface, centerReservoir,
                rng, sparams, gRESTIRGDI.params, selectedSample);
        }

        // Store selected light sample in the output reservoir.
        RESTIR_StoreReservoir(reservoir, gRESTIRGDI.params, pixelID, outReservoirIndex);
    }

    /** Perform temporal resampling.
    */
    void doTemporalResampling(uint2 pixelID, uint2 groupId, uint candidateReservoirIndex, uint priorFrameReservoirIndex, uint outReservoirIndex)
    {
        uint2 pixelPosition = RESTIR_ReservoirToPixelPos(pixelID, gRESTIRGDI.params);

        // Initialize random sampler.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 4);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RESTIR_Reservoir reservoir = {};

        if (RAB_IsSurfaceValid(surface))
        {
            // Load reservoir containing the initial candidate at the current pixel.
            RESTIR_Reservoir candidateReservoir = RESTIR_LoadReservoir(gRESTIRGDI.params, pixelID, candidateReservoirIndex);

            // Setup resampling parameters.
            RESTIR_TemporalResamplingParameters tparams;
            tparams.screenSpaceMotion = float3(gRESTIRGDI.motionVectors[pixelPosition] * gRESTIRGDI.frameDim, 0.f); // TODO handle linear z
            tparams.sourceBufferIndex = priorFrameReservoirIndex;
            tparams.maxHistoryLength = gRESTIRGDI.maxHistoryLength;
            tparams.biasCorrectionMode = gRESTIRGDI.biasCorrectionMode;
            tparams.depthThreshold = gRESTIRGDI.depthThreshold;
            tparams.normalThreshold = gRESTIRGDI.normalThreshold;
            tparams.enableVisibilityShortcut = gRESTIRGDI.enableVisibilityShortcut;
            tparams.enablePermutationSampling = gRESTIRGDI.enablePermutationSampling;

            // Execute resampling.
            RAB_LightSample selectedSample = {};
            int2 selectedSamplePos = 0;
            reservoir = RESTIR_TemporalResampling(pixelPosition, surface, candidateReservoir,
                rng, tparams, gRESTIRGDI.params, selectedSamplePos, selectedSample);
        }

        // If requested (and if we're in a compute shader), spawn a filter to surpress boiling.
#ifdef RESTIR_ENABLE_BOILING_FILTER
        if (gRESTIRGDI.boilingFilterStrength > 0)
        {
            RESTIR_BoilingFilter(groupId, gRESTIRGDI.boilingFilterStrength, gRESTIRGDI.params, reservoir);
        }
#endif

        // Store selected light sample in the output reservoir.
        RESTIR_StoreReservoir(reservoir, gRESTIRGDI.params, pixelID, outReservoirIndex);
    }

    /** Perform spatio-temporal resampling.
    */
    void doSpatiotemporalResampling(uint2 pixelID, uint2 groupId, uint candidateReservoirIndex, uint priorFrameReservoirIndex, uint outReservoirIndex)
    {
        uint2 pixelPosition = RESTIR_ReservoirToPixelPos(pixelID, gRESTIRGDI.params);

        // Initialize random sampler.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 4);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RESTIR_Reservoir reservoir = {};

        if (RAB_IsSurfaceValid(surface))
        {
            // Load reservoir containing the initial candidate at the current pixel.
            RESTIR_Reservoir candidateReservoir = RESTIR_LoadReservoir(gRESTIRGDI.params, pixelID, candidateReservoirIndex);

            // Setup resampling parameters.
            RESTIR_SpatioTemporalResamplingParameters stparams;
            stparams.screenSpaceMotion = float3(gRESTIRGDI.motionVectors[pixelPosition] * gRESTIRGDI.frameDim, 0.f); // TODO handle linear z
            stparams.sourceBufferIndex = priorFrameReservoirIndex;
            stparams.maxHistoryLength = gRESTIRGDI.maxHistoryLength;
            stparams.biasCorrectionMode = gRESTIRGDI.biasCorrectionMode;
            stparams.depthThreshold = gRESTIRGDI.depthThreshold;
            stparams.normalThreshold = gRESTIRGDI.normalThreshold;
            stparams.numSamples = gRESTIRGDI.spatialSampleCount + 1;
            stparams.numDisocclusionBoostSamples = gRESTIRGDI.disocclusionSampleCount;
            stparams.samplingRadius = gRESTIRGDI.samplingRadius;
            stparams.enableVisibilityShortcut = gRESTIRGDI.enableVisibilityShortcut;
            stparams.enablePermutationSampling = gRESTIRGDI.enablePermutationSampling;

            // Execute resampling.
            RAB_LightSample selectedSample = {};
            int2 selectedSamplePos = 0;
            reservoir = RESTIR_SpatioTemporalResampling(pixelPosition, surface, candidateReservoir,
                rng, stparams, gRESTIRGDI.params, selectedSamplePos, selectedSample);
        }

        // If requested (and if we're in a compute shader), spawn a filter to surpress boiling.
#ifdef RESTIR_ENABLE_BOILING_FILTER
        if (gRESTIRGDI.boilingFilterStrength > 0)
        {
            RESTIR_BoilingFilter(groupId, gRESTIRGDI.boilingFilterStrength, gRESTIRGDI.params, reservoir);
        }
#endif

        // Store selected light sample in the output reservoir.
        RESTIR_StoreReservoir(reservoir, gRESTIRGDI.params, pixelID, outReservoirIndex);
    }

    /** Returns the final sample; a more cryptic form of getFinalSample(), from above, exposing RESTIRGDI bridge structures
        \param[in]  pixel Pixel coordinate.
        \param[out] surface An RESTIRGDI RAB_Surface structure at this pixel.
        \param[out] lightSample An RESTIRGDI RAB_LightSample structure for the final selected light.
        \param[out] Li Weighted unoccluded incident radiance (i.e., incident radiance / pdf)
        \return Returns true if a valid sample was returned.
    */
    bool getFinalSample(uint2 pixel, out RAB_Surface surface, out RAB_LightSample lightSample, out float3 Li)
    {
        // Zero initialize out parameters.
        lightSample = {};
        Li = {};

        // Load surface data and reservoir containing the final light sample.
        uint2 pixelPosition = RESTIR_ReservoirToPixelPos(pixel, gRESTIRGDI.params);
        surface = RAB_GetGBufferSurface(pixelPosition, false);
        RESTIR_Reservoir reservoir = RESTIR_LoadReservoir(gRESTIRGDI.params, pixel, gRESTIRGDI.finalShadingReservoir);

        // Abort if we don't have a valid surface or reservoir.
        if (!RAB_IsSurfaceValid(surface) || !RESTIR_IsValidReservoir(reservoir)) return false;

        // Load the selected light and the specific light sample on it.
        uint lightIdx = RESTIR_GetReservoirLightIndex(reservoir);
        float2 lightUV = RESTIR_GetReservoirSampleUV(reservoir);
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);
        lightSample = lightInfo.calcSample(surface.position, lightUV, false); // Make sure to sample full RGB incident radiance.

        // Compute incident radiance pre-divided by the PDF.
        // The incident radiance returned from the light sample is already pre-divided by the PDF.
        // We only need to apply the inverse PDF from the reservoir.
        Li = lightSample.Li * RESTIR_GetReservoirInvPdf(reservoir);

        return true;
    }
}

ParameterBlock<ReSTIR> gRESTIRGDI;
