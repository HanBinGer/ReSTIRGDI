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
#include "ReSTIRGDI.h"
#include "Core/Error.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Timing/Profiler.h"
#include <fstd/bit.h> // TODO C++20: Replace with <bit>
#include <cassert>
#include <algorithm>
#include <vector>
#include <numeric>
#include <math.h>

constexpr float c_pi = 3.1415926535f;

static bool IsNonzeroPowerOf2(uint32_t i)
{
    return ((i & (i - 1)) == 0) && (i > 0);
}

namespace restir
{
Context::Context(const ContextParameters& params) : m_Params(params)
{
    assert(IsNonzeroPowerOf2(params.TileSize));
    assert(IsNonzeroPowerOf2(params.TileCount));
    assert(params.RenderWidth > 0);
    assert(params.RenderHeight > 0);

    uint32_t renderWidth = (params.CheckerboardSamplingMode == CheckerboardMode::Off) ? params.RenderWidth : (params.RenderWidth + 1) / 2;
    uint32_t renderWidthBlocks = (renderWidth + RESTIR_RESERVOIR_BLOCK_SIZE - 1) / RESTIR_RESERVOIR_BLOCK_SIZE;
    uint32_t renderHeightBlocks = (params.RenderHeight + RESTIR_RESERVOIR_BLOCK_SIZE - 1) / RESTIR_RESERVOIR_BLOCK_SIZE;
    m_ReservoirBlockRowPitch = renderWidthBlocks * (RESTIR_RESERVOIR_BLOCK_SIZE * RESTIR_RESERVOIR_BLOCK_SIZE);
    m_ReservoirArrayPitch = m_ReservoirBlockRowPitch * renderHeightBlocks;

    m_RegirCellOffset = m_Params.TileCount * m_Params.TileSize;

    ComputeOnionJitterCurve();
}

static float3 SphericalToCartesian(const float radius, const float azimuth, const float elevation)
{
    return float3{radius * cosf(azimuth) * cosf(elevation), radius * sinf(elevation), radius * sinf(azimuth) * cosf(elevation)};
}

static float Distance(const float3& a, const float3& b)
{
    float3 d{a.x - b.x, a.y - b.y, a.z - b.z};
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

void Context::ComputeOnionJitterCurve()
{
    std::vector<float> cubicRootFactors;
    std::vector<float> linearFactors;

    int layerGroupIndex = 0;
    for (const auto& layerGroup : m_OnionLayers)
    {
        for (int layerIndex = 0; layerIndex < layerGroup.layerCount; layerIndex++)
        {
            const float innerRadius = layerGroup.innerRadius * powf(layerGroup.layerScale, float(layerIndex));
            const float outerRadius = innerRadius * layerGroup.layerScale;
            const float middleRadius = (innerRadius + outerRadius) * 0.5f;
            float maxCellRadius = 0.f;

            for (int ringIndex = 0; ringIndex < layerGroup.ringCount; ringIndex++)
            {
                const auto& ring = m_OnionRings[layerGroup.ringOffset + ringIndex];

                const float middleElevation = layerGroup.equatorialCellAngle * float(ringIndex);
                const float vertexElevation =
                    (ringIndex == 0) ? layerGroup.equatorialCellAngle * 0.5f : middleElevation - layerGroup.equatorialCellAngle * 0.5f;

                const float middleAzimuth = 0.f;
                const float vertexAzimuth = ring.cellAngle;

                const float3 middlePoint = SphericalToCartesian(middleRadius, middleAzimuth, middleElevation);
                const float3 vertexPoint = SphericalToCartesian(outerRadius, vertexAzimuth, vertexElevation);

                const float cellRadius = Distance(middlePoint, vertexPoint);

                maxCellRadius = std::max(maxCellRadius, cellRadius);
            }

#if PRINT_JITTER_CURVE
            char buf[256];
            sprintf_s(buf, "%.3f,%.3f\n", middleRadius, maxCellRadius);
            OutputDebugStringA(buf);
#endif

            if (layerGroupIndex < int(m_OnionLayers.size()) - 1)
            {
                float cubicRootFactor = maxCellRadius * powf(middleRadius, -1.f / 3.f);
                cubicRootFactors.push_back(cubicRootFactor);
            }
            else
            {
                float linearFactor = maxCellRadius / middleRadius;
                linearFactors.push_back(linearFactor);
            }
        }

        layerGroupIndex++;
    }

    // Compute the median of the cubic root factors, there are some outliers in the curve
    if (!cubicRootFactors.empty())
    {
        std::sort(cubicRootFactors.begin(), cubicRootFactors.end());
        m_OnionCubicRootFactor = cubicRootFactors[cubicRootFactors.size() / 2];
    }
    else
    {
        m_OnionCubicRootFactor = 0.f;
    }

    // Compute the average of the linear factors, they're all the same anyway
    float sumOfLinearFactors = std::accumulate(linearFactors.begin(), linearFactors.end(), 0.f);
    m_OnionLinearFactor = sumOfLinearFactors / std::max(float(linearFactors.size()), 1.f);
}

const ContextParameters& Context::GetParameters() const
{
    return m_Params;
}

uint32_t Context::GetRisBufferElementCount() const
{
    uint32_t size = 0;
    size += m_Params.TileCount * m_Params.TileSize;
    size += m_Params.EnvironmentTileCount * m_Params.EnvironmentTileSize;

    return size;
}

uint32_t Context::GetReservoirBufferElementCount() const
{
    return m_ReservoirArrayPitch;
}

// 32 bit Jenkins hash
static uint32_t JenkinsHash(uint32_t a)
{
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

void Context::FillRuntimeParameters(RESTIR_ResamplingRuntimeParameters& runtimeParams, const FrameParameters& frame) const
{
    runtimeParams.firstLocalLight = frame.firstLocalLight;
    runtimeParams.numLocalLights = frame.numLocalLights;
    runtimeParams.firstInfiniteLight = frame.firstInfiniteLight;
    runtimeParams.numInfiniteLights = frame.numInfiniteLights;
    runtimeParams.environmentLightPresent = frame.environmentLightPresent;
    runtimeParams.environmentLightIndex = frame.environmentLightIndex;
    runtimeParams.neighborOffsetMask = m_Params.NeighborOffsetCount - 1;
    runtimeParams.tileSize = m_Params.TileSize;
    runtimeParams.tileCount = m_Params.TileCount;
    runtimeParams.enableLocalLightImportanceSampling = frame.enableLocalLightImportanceSampling;
    runtimeParams.reservoirBlockRowPitch = m_ReservoirBlockRowPitch;
    runtimeParams.reservoirArrayPitch = m_ReservoirArrayPitch;
    runtimeParams.environmentRisBufferOffset = m_RegirCellOffset;
    runtimeParams.environmentTileCount = m_Params.EnvironmentTileCount;
    runtimeParams.environmentTileSize = m_Params.EnvironmentTileSize;
    runtimeParams.uniformRandomNumber = JenkinsHash(frame.frameIndex);
    runtimeParams.pad1 = 0;
    runtimeParams.pad2 = 0;
    runtimeParams.pad3 = 0;

    switch (m_Params.CheckerboardSamplingMode)
    {
    case CheckerboardMode::Black:
        runtimeParams.activeCheckerboardField = (frame.frameIndex & 1) ? 1 : 2;
        break;
    case CheckerboardMode::White:
        runtimeParams.activeCheckerboardField = (frame.frameIndex & 1) ? 2 : 1;
        break;
    default:
        runtimeParams.activeCheckerboardField = 0;
    }
}

void Context::FillNeighborOffsetBuffer(uint8_t* buffer) const
{
    // Create a sequence of low-discrepancy samples within a unit radius around the origin
    // for "randomly" sampling neighbors during spatial resampling

    int R = 250;
    const float phi2 = 1.0f / 1.3247179572447f;
    uint32_t num = 0;
    float u = 0.5f;
    float v = 0.5f;
    while (num < m_Params.NeighborOffsetCount * 2)
    {
        u += phi2;
        v += phi2 * phi2;
        if (u >= 1.0f)
            u -= 1.0f;
        if (v >= 1.0f)
            v -= 1.0f;

        float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
        if (rSq > 0.25f)
            continue;

        buffer[num++] = int8_t((u - 0.5f) * R);
        buffer[num++] = int8_t((v - 0.5f) * R);
    }
}

void ComputePdfTextureSize(uint32_t maxItems, uint32_t& outWidth, uint32_t& outHeight, uint32_t& outMipLevels)
{
    // Compute the size of a power-of-2 rectangle that fits all items, 1 item per pixel
    double textureWidth = std::max(1.0, ceil(sqrt(double(maxItems))));
    textureWidth = exp2(ceil(log2(textureWidth)));
    double textureHeight = std::max(1.0, ceil(maxItems / textureWidth));
    textureHeight = exp2(ceil(log2(textureHeight)));
    double textureMips = std::max(1.0, log2(std::max(textureWidth, textureHeight)));

    outWidth = uint32_t(textureWidth);
    outHeight = uint32_t(textureHeight);
    outMipLevels = uint32_t(textureMips);
}
} // namespace restir

namespace Falcor
{
namespace
{
// Shader location
const std::string kReflectTypesShaderFile = "Rendering/ReSTIRGDI/ReflectTypes.cs.slang";
const std::string kReSTIRGDIShadersFile = "Rendering/ReSTIRGDI/ReSTIRGDISetup.cs.slang";
const std::string kLightUpdaterShaderFile = "Rendering/ReSTIRGDI/LightUpdater.cs.slang";
const std::string kEnvLightUpdaterShaderFile = "Rendering/ReSTIRGDI/EnvLightUpdater.cs.slang";

/** Config setting : Maximum number of unique screen-sized reservoir bufers needed by any
    ReSTIRGDI pipelines we create in this pass. Just controls memory allocation (not really perf).
*/
const uint32_t kMaxReservoirs = 3;        ///< Number of reservoirs per pixel to allocate (and hence the max # used).
const uint32_t kCandidateReservoirID = 2; ///< We always store per-frame candidate lights in reservoir #2.

const uint32_t kMinPresampledTileCount = 1;
const uint32_t kMaxPresampledTileCount = 1024;

const uint32_t kMinPresampledTileSize = 256;
const uint32_t kMaxPresampledTileSize = 8192;

const uint32_t kMinLightCandidateCount = 0;
const uint32_t kMaxLightCandidateCount = 256;

const float kMinSpatialRadius = 0.f;
const float kMaxSpatialRadius = 50.f;

const uint32_t kMinSpatialSampleCount = 0;
const uint32_t kMaxSpatialSampleCount = 25;

const uint32_t kMinSpatialIterations = 0;
const uint32_t kMaxSpatialIterations = 10;

const uint32_t kMinMaxHistoryLength = 0;
const uint32_t kMaxMaxHistoryLength = 50;

template<typename T>
void validateRange(T& value, T minValue, T maxValue, const char* name)
{
    if (value < minValue || value > maxValue)
    {
        logWarning("ReSTIRGDI: '{}' is {}. Clamping to [{},{}].", name, value, minValue, maxValue);
        value = std::clamp(value, minValue, maxValue);
    }
};
} // namespace

ReSTIRGDI::ReSTIRGDI(ref<IScene> pScene, const Options& options)
    : mpScene(std::move(pScene)), mpDevice(mpScene->getDevice()), mOptions(options)
{
    FALCOR_ASSERT(mpScene);
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("ReSTIRGDI requires Shader Model 6.5 support.");

    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

    mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });

    setOptions(options);
}

void ReSTIRGDI::setOptions(const Options& options)
{
    Options newOptions = options;

    validateRange(newOptions.presampledTileCount, kMinPresampledTileCount, kMaxPresampledTileCount, "presampledTileCount");
    validateRange(newOptions.presampledTileSize, kMinPresampledTileSize, kMaxPresampledTileSize, "presampledTileSize");

    validateRange(newOptions.localLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "localLightCandidateCount");
    validateRange(newOptions.infiniteLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "infiniteLightCandidateCount");
    validateRange(newOptions.envLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "envLightCandidateCount");
    validateRange(newOptions.brdfCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "brdfCandidateCount");
    validateRange(newOptions.brdfCutoff, 0.f, 1.f, "brdfCutoff");

    validateRange(newOptions.depthThreshold, 0.f, 1.f, "depthThreshold");
    validateRange(newOptions.normalThreshold, 0.f, 1.f, "normalThreshold");

    validateRange(newOptions.samplingRadius, kMinSpatialRadius, kMaxSpatialRadius, "samplingRadius");
    validateRange(newOptions.spatialSampleCount, kMinSpatialSampleCount, kMaxSpatialSampleCount, "spatialSampleCount");
    validateRange(newOptions.spatialIterations, kMinSpatialIterations, kMaxSpatialIterations, "spatialIterations");

    validateRange(newOptions.maxHistoryLength, kMinMaxHistoryLength, kMaxMaxHistoryLength, "maxHistoryLength");
    validateRange(newOptions.boilingFilterStrength, 0.f, 1.f, "boilingFilterStrength");

    if (newOptions.mode != mOptions.mode)
    {
        mFlags.clearReservoirs = true;
        mLastFrameReservoirID = 0; // Switching out of Talbot mode can break without this.
    }

    if (newOptions.presampledTileCount != mOptions.presampledTileCount || newOptions.presampledTileSize != mOptions.presampledTileSize)
    {
        mpRESTIRContext = nullptr;
    }

    if (newOptions.envLightCandidateCount != mOptions.envLightCandidateCount && newOptions.envLightCandidateCount == 0)
    {
        // Avoid fadeout when disabling env sampling
        mFlags.clearReservoirs = true;
    }

    if (newOptions.testCandidateVisibility != mOptions.testCandidateVisibility)
    {
        mFlags.clearReservoirs = true;
    }

    mOptions = newOptions;
}

DefineList ReSTIRGDI::getDefines() const
{
    DefineList defines;
    defines.add("ReSTIRGDI_INSTALLED", "1");
    return defines;
}

void ReSTIRGDI::bindShaderData(const ShaderVar& rootVar)
{
    bindShaderDataInternal(rootVar, nullptr, false);
}

void ReSTIRGDI::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
{
    // Check for scene changes that require shader recompilation.
    // TODO: We may want to reset other data that depends on the scene geometry or materials.
    if (is_set(mUpdateFlags, IScene::UpdateFlags::RecompileNeeded) || is_set(mUpdateFlags, IScene::UpdateFlags::GeometryChanged))
    {
        mFlags.recompileShaders = true;
    }

    // Make sure the light collection is created.
    mpScene->getILightCollection(pRenderContext);

    // Initialize previous frame camera data.
    if (mFrameIndex == 0)
        mPrevCameraData = mpScene->getCamera()->getData();

    // Update the screen resolution.
    if (any(frameDim != mFrameDim))
    {
        mFrameDim = frameDim;
        // Resizes require reallocating resources.
        mpRESTIRContext = nullptr;
    }

    // Load shaders if required.
    if (mFlags.recompileShaders)
        loadShaders();

    // Create ReSTIRGDI context and allocate resources if required.
    if (!mpRESTIRContext)
        prepareResources(pRenderContext);

    // Clear reservoir buffer if requested. This can be required when changing configuration options.
    if (mFlags.clearReservoirs)
    {
        pRenderContext->clearUAV(mpReservoirBuffer->getUAV().get(), uint4(0));
        mFlags.clearReservoirs = false;
    }

    // Determine what, if anything happened since last frame. TODO: Make more granular / flexible.

    // Emissive lights.
    if (is_set(mUpdateFlags, IScene::UpdateFlags::LightCollectionChanged))
        mFlags.updateEmissiveLights = true;
    if (is_set(mUpdateFlags, IScene::UpdateFlags::EmissiveMaterialsChanged))
        mFlags.updateEmissiveLightsFlux = true;
    // Analytic lights.
    if (is_set(mUpdateFlags, IScene::UpdateFlags::LightCountChanged))
        mFlags.updateAnalyticLights = true;
    if (is_set(mUpdateFlags, IScene::UpdateFlags::LightPropertiesChanged))
        mFlags.updateAnalyticLights = true;
    if (is_set(mUpdateFlags, IScene::UpdateFlags::LightIntensityChanged))
        mFlags.updateAnalyticLightsFlux = true;
    // Env light. Update the env light PDF either if the env map changed or its tint/intensity changed.
    if (is_set(mUpdateFlags, IScene::UpdateFlags::EnvMapChanged))
        mFlags.updateEnvLight = true;
    if (is_set(mUpdateFlags, IScene::UpdateFlags::EnvMapPropertiesChanged) &&
        is_set(mpScene->getEnvMap()->getChanges(), EnvMap::Changes::Intensity))
        mFlags.updateEnvLight = true;

    if (is_set(mUpdateFlags, IScene::UpdateFlags::RenderSettingsChanged))
    {
        mFlags.updateAnalyticLights = true;
        mFlags.updateAnalyticLightsFlux = true;
        mFlags.updateEmissiveLights = true;
        mFlags.updateEmissiveLightsFlux = true;
        mFlags.updateEnvLight = true;
    }

    mUpdateFlags = IScene::UpdateFlags::None;

    mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
}

void ReSTIRGDI::endFrame(RenderContext* pRenderContext)
{
    // Increment our frame counter and swap surface buffers.
    mFrameIndex++;
    mCurrentSurfaceBufferIndex = 1 - mCurrentSurfaceBufferIndex;

    // Remember this frame's camera data for use next frame.
    mPrevCameraData = mpScene->getCamera()->getData();

    mpPixelDebug->endFrame(pRenderContext);
}

void ReSTIRGDI::update(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIRGDI::update");

    // Create a PDF texture for our primitive lights (for now, just triangles)
    updateLights(pRenderContext);
    updateEnvLight(pRenderContext);

    // Update our parameters for the current frame and pass them into our GPU structure.
    setReSTIRGDIFrameParameters();

    // Create tiles of presampled lights once per frame to improve per-pixel memory coherence.
    presampleLights(pRenderContext);

    // Reservoir buffer containing reservoirs after sampling/resampling.
    uint32_t outputReservoirID;

    switch (mOptions.mode)
    {
    case Mode::NoResampling:
        generateCandidates(pRenderContext, kCandidateReservoirID);
        outputReservoirID = kCandidateReservoirID;
        break;
    case Mode::SpatialResampling:
        generateCandidates(pRenderContext, kCandidateReservoirID);
        testCandidateVisibility(pRenderContext, kCandidateReservoirID);
        outputReservoirID = spatialResampling(pRenderContext, kCandidateReservoirID);
        break;
    case Mode::TemporalResampling:
        generateCandidates(pRenderContext, kCandidateReservoirID);
        testCandidateVisibility(pRenderContext, kCandidateReservoirID);
        outputReservoirID = temporalResampling(pRenderContext, pMotionVectors, kCandidateReservoirID, mLastFrameReservoirID);
        break;
    case Mode::SpatiotemporalResampling:
        generateCandidates(pRenderContext, kCandidateReservoirID);
        testCandidateVisibility(pRenderContext, kCandidateReservoirID);
        outputReservoirID = spatiotemporalResampling(pRenderContext, pMotionVectors, kCandidateReservoirID, mLastFrameReservoirID);
        break;
    }

    // Remember output reservoir buffer for the next frame (and shading this frame).
    mLastFrameReservoirID = outputReservoirID;
}

void ReSTIRGDI::bindShaderDataInternal(const ShaderVar& rootVar, const ref<Texture>& pMotionVectors, bool bindScene)
{
    auto var = rootVar["gRESTIRGDI"];

    // Send our parameter structure down
    var["params"].setBlob(&mRESTIRShaderParams, sizeof(mRESTIRShaderParams));

    // Parameters needed inside the core ReSTIRGDI application bridge
    var["frameIndex"] = mFrameIndex;
    var["rayEpsilon"] = mOptions.rayEpsilon;
    var["frameDim"] = mFrameDim;
    var["pixelCount"] = mFrameDim.x * mFrameDim.y;
    var["storeCompactLightInfo"] = mOptions.storeCompactLightInfo;
    var["useEmissiveTextures"] = mOptions.useEmissiveTextures;
    var["currentSurfaceBufferIndex"] = mCurrentSurfaceBufferIndex;
    var["prevSurfaceBufferIndex"] = 1 - mCurrentSurfaceBufferIndex;

    // Parameters for initial candidate samples
    var["localLightCandidateCount"] = mOptions.localLightCandidateCount;
    var["infiniteLightCandidateCount"] = mOptions.infiniteLightCandidateCount;
    var["envLightCandidateCount"] = mOptions.envLightCandidateCount;
    var["brdfCandidateCount"] = mOptions.brdfCandidateCount;

    // Parameters for general sample reuse
    var["maxHistoryLength"] = mOptions.maxHistoryLength;
    var["biasCorrectionMode"] = uint(mOptions.biasCorrection);

    // Parameter for final shading
    var["finalShadingReservoir"] = mLastFrameReservoirID;

    // Parameters for generally spatial sample reuse
    var["spatialSampleCount"] = mOptions.spatialSampleCount;
    var["disocclusionSampleCount"] = mOptions.spatialSampleCount;
    var["samplingRadius"] = mOptions.samplingRadius;
    var["depthThreshold"] = mOptions.depthThreshold;
    var["normalThreshold"] = mOptions.normalThreshold;
    var["boilingFilterStrength"] = mOptions.boilingFilterStrength;
    var["enableVisibilityShortcut"] = mOptions.enableVisibilityShortcut;
    var["enablePermutationSampling"] = mOptions.enablePermutationSampling;

    // Parameters for last frame's camera coordinate
    var["prevCameraU"] = mPrevCameraData.cameraU;
    var["prevCameraV"] = mPrevCameraData.cameraV;
    var["prevCameraW"] = mPrevCameraData.cameraW;
    var["prevCameraJitter"] = float2(mPrevCameraData.jitterX, mPrevCameraData.jitterY);

    // Setup textures and other buffers needed by the ReSTIRGDI bridge
    var["lightInfo"] = mpLightInfoBuffer;
    var["surfaceData"] = mpSurfaceDataBuffer;
    var["risBuffer"] = mpLightTileBuffer;
    var["compactLightInfo"] = mpCompactLightInfoBuffer;
    var["reservoirs"] = mpReservoirBuffer;
    var["neighborOffsets"] = mpNeighborOffsetsBuffer;
    var["motionVectors"] = pMotionVectors;

    // PDF textures for importance sampling. Some shaders need UAVs, some SRVs
    var["localLightPdfTexture"] = mpLocalLightPdfTexture;
    var["envLightLuminanceTexture"] = mpEnvLightLuminanceTexture;
    var["envLightPdfTexture"] = mpEnvLightPdfTexture;

    // Bind the scene.
    if (bindScene)
        mpScene->bindShaderData(rootVar["gScene"]);
}

void ReSTIRGDI::updateLights(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "updateLights");

    // First, update our list of analytic lights to use.
    if (mFlags.updateAnalyticLights)
    {
        if (mpScene->useAnalyticLights())
        {
            std::vector<uint32_t> localAnalyticLightIDs;
            std::vector<uint32_t> infiniteAnalyticLightIDs;

            for (uint32_t lightID = 0; lightID < mpScene->getActiveAnalyticLights().size(); ++lightID)
            {
                const auto& pLight = mpScene->getActiveAnalyticLights()[lightID];
                switch (pLight->getType())
                {
                case LightType::Point:
                    localAnalyticLightIDs.push_back(lightID);
                    break;
                case LightType::Directional:
                case LightType::Distant:
                    infiniteAnalyticLightIDs.push_back(lightID);
                    break;
                case LightType::Rect:
                case LightType::Disc:
                case LightType::Sphere:
                    // We currently ignore all analytic area lights.
                    break;
                default:
                    break;
                }
            }

            // Update light counts.
            mLights.localAnalyticLightCount = (uint32_t)localAnalyticLightIDs.size();
            mLights.infiniteAnalyticLightCount = (uint32_t)infiniteAnalyticLightIDs.size();

            // Update list of light IDs, local lights followed by infinite lights.
            mLights.analyticLightIDs.clear();
            mLights.analyticLightIDs.reserve(localAnalyticLightIDs.size() + infiniteAnalyticLightIDs.size());
            mLights.analyticLightIDs.insert(mLights.analyticLightIDs.end(), localAnalyticLightIDs.begin(), localAnalyticLightIDs.end());
            mLights.analyticLightIDs.insert(
                mLights.analyticLightIDs.end(), infiniteAnalyticLightIDs.begin(), infiniteAnalyticLightIDs.end()
            );

            // Create GPU buffer for holding light IDs.
            if (!mLights.analyticLightIDs.empty() &&
                (!mpAnalyticLightIDBuffer || mpAnalyticLightIDBuffer->getElementCount() < mLights.analyticLightIDs.size()))
            {
                mpAnalyticLightIDBuffer = mpDevice->createStructuredBuffer(sizeof(uint32_t), (uint32_t)mLights.analyticLightIDs.size());
            }

            // Update GPU buffer.
            if (mpAnalyticLightIDBuffer)
                mpAnalyticLightIDBuffer->setBlob(mLights.analyticLightIDs.data(), 0, mLights.analyticLightIDs.size() * sizeof(uint32_t));
        }
        else
        {
            // Analytic lights are disabled.
            mLights.localAnalyticLightCount = 0;
            mLights.infiniteAnalyticLightCount = 0;
            mLights.analyticLightIDs.clear();
        }
    }

    // Update other light counts.
    mLights.emissiveLightCount =
        mpScene->useEmissiveLights() ? mpScene->getILightCollection(pRenderContext)->getActiveLightCount(pRenderContext) : 0;
    mLights.envLightPresent = mpScene->useEnvLight();

    uint32_t localLightCount = mLights.getLocalLightCount();
    uint32_t totalLightCount = mLights.getTotalLightCount();

    // Allocate buffer for light infos.
    if (!mpLightInfoBuffer || mpLightInfoBuffer->getElementCount() < totalLightCount)
    {
        mpLightInfoBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["lightInfo"], totalLightCount);
    }

    // Allocate local light PDF texture, which ReSTIRGDI uses for importance sampling.
    {
        uint32_t width, height, mipLevels;
        restir::ComputePdfTextureSize(localLightCount, width, height, mipLevels);
        if (!mpLocalLightPdfTexture || mpLocalLightPdfTexture->getWidth() != width || mpLocalLightPdfTexture->getHeight() != height ||
            mpLocalLightPdfTexture->getMipCount() != mipLevels)
        {
            mpLocalLightPdfTexture = mpDevice->createTexture2D(
                width,
                height,
                ResourceFormat::R16Float,
                1,
                mipLevels,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget
            );
        }
    }

    // If the layout of local lights has changed, we need to make sure to remove any extra non-zero entries in the local light PDF texture.
    // We simply clear the texture and populate it from scratch.
    if (mLights.prevEmissiveLightCount != mLights.emissiveLightCount ||
        mLights.prevLocalAnalyticLightCount != mLights.localAnalyticLightCount)
    {
        mFlags.updateAnalyticLightsFlux = true;
        mFlags.updateEmissiveLightsFlux = true;
        pRenderContext->clearUAV(mpLocalLightPdfTexture->getUAV().get(), float4(0.f));
    }

    // If the number of emissive lights has changed, we need to update the analytic lights because they change position in the light info
    // buffer.
    if (mLights.prevEmissiveLightCount != mLights.emissiveLightCount)
    {
        mFlags.updateAnalyticLights = true;
    }

    // Run the update pass if any lights have changed.
    if (mFlags.updateEmissiveLights || mFlags.updateEmissiveLightsFlux || mFlags.updateAnalyticLights || mFlags.updateAnalyticLightsFlux ||
        mFlags.updateEnvLight)
    {
        // Compute launch dimensions.
        uint2 threadCount = {8192u, div_round_up(totalLightCount, 8192u)};

        auto var = mpUpdateLightsPass->getRootVar()["gLightUpdater"];
        var["lightInfo"] = mpLightInfoBuffer;
        var["localLightPdf"] = mpLocalLightPdfTexture;
        var["analyticLightIDs"] = mpAnalyticLightIDBuffer;
        var["threadCount"] = threadCount;
        var["totalLightCount"] = mLights.getTotalLightCount();
        var["firstLocalAnalyticLight"] = mLights.emissiveLightCount;
        var["firstInfiniteAnalyticLight"] = mLights.emissiveLightCount + mLights.localAnalyticLightCount;
        var["envLightIndex"] = mLights.getEnvLightIndex();
        var["updateEmissiveLights"] = mFlags.updateEmissiveLights;
        var["updateEmissiveLightsFlux"] = mFlags.updateEmissiveLightsFlux;
        var["updateAnalyticLights"] = mFlags.updateAnalyticLights;
        var["updateAnalyticLightsFlux"] = mFlags.updateAnalyticLightsFlux;
        mpScene->bindShaderData(mpUpdateLightsPass->getRootVar()["gScene"]);
        mpUpdateLightsPass->execute(pRenderContext, threadCount.x, threadCount.y);
    }

    // Update the light PDF texture mipmap chain if necessary.
    if (mFlags.updateEmissiveLightsFlux | mFlags.updateAnalyticLightsFlux)
    {
        mpLocalLightPdfTexture->generateMips(pRenderContext);
    }

    // Keep track of the number of local lights for the next frame.
    mLights.prevEmissiveLightCount = mLights.emissiveLightCount;
    mLights.prevLocalAnalyticLightCount = mLights.localAnalyticLightCount;

    mFlags.updateEmissiveLights = false;
    mFlags.updateEmissiveLightsFlux = false;
    mFlags.updateAnalyticLights = false;
    mFlags.updateAnalyticLightsFlux = false;
}

void ReSTIRGDI::updateEnvLight(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "updateEnvLight");

    // If scene uses an environment light, create a luminance & pdf texture for sampling it.
    if (mpScene->useEnvLight() && mFlags.updateEnvLight)
    {
        const auto& pEnvMap = mpScene->getEnvMap()->getEnvMap();
        FALCOR_ASSERT(pEnvMap);
        auto& pLuminanceTexture = mpEnvLightLuminanceTexture;
        auto& pPdfTexture = mpEnvLightPdfTexture;

        // ReSTIRGDI expects power-of-two textures.
        uint32_t width = fstd::bit_ceil(pEnvMap->getWidth());
        uint32_t height = fstd::bit_ceil(pEnvMap->getHeight());

        // Create luminance texture if it doesn't exist yet or has the wrong dimensions.
        if (!pLuminanceTexture || pLuminanceTexture->getWidth() != width || pLuminanceTexture->getHeight() != height)
        {
            pLuminanceTexture = mpDevice->createTexture2D(
                width,
                height,
                ResourceFormat::R32Float,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget
            );
        }

        // Create pdf texture if it doesn't exist yet or has the wrong dimensions.
        if (!pPdfTexture || pPdfTexture->getWidth() != width || pPdfTexture->getHeight() != height)
        {
            pPdfTexture = mpDevice->createTexture2D(
                width,
                height,
                ResourceFormat::R32Float,
                1,
                Resource::kMaxPossible,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget
            );
        }

        // Update env light textures.
        auto var = mpUpdateEnvLightPass->getRootVar()["gEnvLightUpdater"];
        var["envLightLuminance"] = pLuminanceTexture;
        var["envLightPdf"] = pPdfTexture;
        var["texDim"] = uint2(width, height);
        mpScene->bindShaderData(mpUpdateEnvLightPass->getRootVar()["gScene"]);
        mpUpdateEnvLightPass->execute(pRenderContext, width, height);

        // Create a mipmap chain for pdf texure.
        pPdfTexture->generateMips(pRenderContext);
    }

    mFlags.updateEnvLight = false;
}

void ReSTIRGDI::presampleLights(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "presampleLights");

    // Presample local lights.
    {
        auto var = mpPresampleLocalLightsPass->getRootVar();
        bindShaderDataInternal(var, nullptr);
        mpPresampleLocalLightsPass->execute(pRenderContext, mRESTIRContextParams.TileSize, mRESTIRContextParams.TileCount);
    }

    // Presample environment light.
    if (mLights.envLightPresent)
    {
        auto var = mpPresampleEnvLightPass->getRootVar();
        bindShaderDataInternal(var, nullptr);
        mpPresampleEnvLightPass->execute(
            pRenderContext, mRESTIRContextParams.EnvironmentTileSize, mRESTIRContextParams.EnvironmentTileCount
        );
    }
}

void ReSTIRGDI::generateCandidates(RenderContext* pRenderContext, uint32_t outputReservoirID)
{
    FALCOR_PROFILE(pRenderContext, "generateCandidates");

    auto var = mpGenerateCandidatesPass->getRootVar();
    mpPixelDebug->prepareProgram(mpGenerateCandidatesPass->getProgram(), var);

    var["CB"]["gOutputReservoirID"] = outputReservoirID;
    bindShaderDataInternal(var, nullptr);
    mpGenerateCandidatesPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void ReSTIRGDI::testCandidateVisibility(RenderContext* pRenderContext, uint32_t candidateReservoirID)
{
    if (!mOptions.testCandidateVisibility)
        return;

    FALCOR_PROFILE(pRenderContext, "testCandidateVisibility");

    auto var = mpTestCandidateVisibilityPass->getRootVar();
    var["CB"]["gOutputReservoirID"] = candidateReservoirID;
    bindShaderDataInternal(var, nullptr);
    mpTestCandidateVisibilityPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

uint32_t ReSTIRGDI::spatialResampling(RenderContext* pRenderContext, uint32_t inputReservoirID)
{
    FALCOR_PROFILE(pRenderContext, "spatialResampling");

    // We ping-pong between reservoir buffers, depending on # of spatial iterations.
    uint32_t inputID = inputReservoirID;
    uint32_t outputID = (inputID != 1) ? 1 : 0;

    auto var = mpSpatialResamplingPass->getRootVar();
    mpPixelDebug->prepareProgram(mpSpatialResamplingPass->getProgram(), var);

    for (uint32_t i = 0; i < mOptions.spatialIterations; ++i)
    {
        var["CB"]["gInputReservoirID"] = inputID;
        var["CB"]["gOutputReservoirID"] = outputID;
        bindShaderDataInternal(var, nullptr);
        mpSpatialResamplingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        // Ping pong our input and output buffers. (Generally between reservoirs 0 & 1).
        std::swap(inputID, outputID);
    }

    // Return the ID of the last buffer written into.
    return inputID;
}

uint32_t ReSTIRGDI::temporalResampling(
    RenderContext* pRenderContext,
    const ref<Texture>& pMotionVectors,
    uint32_t candidateReservoirID,
    uint32_t lastFrameReservoirID
)
{
    FALCOR_PROFILE(pRenderContext, "temporalResampling");

    // This toggles between storing each frame's outputs between reservoirs 0 and 1.
    uint32_t outputReservoirID = 1 - lastFrameReservoirID;

    auto var = mpTemporalResamplingPass->getRootVar();
    mpPixelDebug->prepareProgram(mpTemporalResamplingPass->getProgram(), var);

    var["CB"]["gTemporalReservoirID"] = lastFrameReservoirID;
    var["CB"]["gInputReservoirID"] = candidateReservoirID;
    var["CB"]["gOutputReservoirID"] = outputReservoirID;
    bindShaderDataInternal(var, pMotionVectors);
    mpTemporalResamplingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

    return outputReservoirID;
}

uint32_t ReSTIRGDI::spatiotemporalResampling(
    RenderContext* pRenderContext,
    const ref<Texture>& pMotionVectors,
    uint32_t candidateReservoirID,
    uint32_t lastFrameReservoirID
)
{
    FALCOR_PROFILE(pRenderContext, "spatiotemporalResampling");

    // This toggles between storing each frame's outputs between reservoirs 0 and 1.
    uint32_t outputReservoirID = 1 - lastFrameReservoirID;

    auto var = mpSpatiotemporalResamplingPass->getRootVar();
    mpPixelDebug->prepareProgram(mpSpatiotemporalResamplingPass->getProgram(), var);

    var["CB"]["gTemporalReservoirID"] = lastFrameReservoirID;
    var["CB"]["gInputReservoirID"] = candidateReservoirID;
    var["CB"]["gOutputReservoirID"] = outputReservoirID;
    bindShaderDataInternal(var, pMotionVectors);
    mpSpatiotemporalResamplingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

    return outputReservoirID;
}

void ReSTIRGDI::loadShaders()
{
    FALCOR_ASSERT(mpScene);
    mpReflectTypes = ComputePass::create(mpDevice, kReflectTypesShaderFile);

    // Issue warnings if packed types are not aligned to 16B for best performance.
    auto pReflector = mpReflectTypes->getProgram()->getReflector();
    FALCOR_ASSERT(pReflector->findType("PackedPolymorphicLight"));
    FALCOR_ASSERT(pReflector->findType("PackedSurfaceData"));
    if (pReflector->findType("PackedPolymorphicLight")->getByteSize() % 16 != 0)
        logWarning("PackedPolymorphicLight struct size is not a multiple of 16B.");
    if (pReflector->findType("PackedSurfaceData")->getByteSize() % 16 != 0)
        logWarning("PackedSurfaceData struct size is not a multiple of 16B.");

    // Helper for creating compute passes.
    auto createComputePass = [&](const std::string& file, const std::string& entryPoint)
    {
        DefineList defines;
        mpScene->getShaderDefines(defines);
        defines.add("ReSTIRGDI_INSTALLED", "1");

        ProgramDesc desc;
        mpScene->getShaderModules(desc.shaderModules);
        desc.addShaderLibrary(file);
        desc.csEntry(entryPoint);
        mpScene->getTypeConformances(desc.typeConformances);
        ref<ComputePass> pPass = ComputePass::create(mpDevice, desc, defines);
        return pPass;
    };

    // Load compute passes for setting up ReSTIRGDI light information.
    mpUpdateLightsPass = createComputePass(kLightUpdaterShaderFile, "main");
    mpUpdateEnvLightPass = createComputePass(kEnvLightUpdaterShaderFile, "main");

    // Load compute passes for ReSTIRGDI sampling and resampling.
    mpPresampleLocalLightsPass = createComputePass(kReSTIRGDIShadersFile, "presampleLocalLights");
    mpPresampleEnvLightPass = createComputePass(kReSTIRGDIShadersFile, "presampleEnvLight");
    mpGenerateCandidatesPass = createComputePass(kReSTIRGDIShadersFile, "generateCandidates");
    mpTestCandidateVisibilityPass = createComputePass(kReSTIRGDIShadersFile, "testCandidateVisibility");
    mpSpatialResamplingPass = createComputePass(kReSTIRGDIShadersFile, "spatialResampling");
    mpTemporalResamplingPass = createComputePass(kReSTIRGDIShadersFile, "temporalResampling");
    mpSpatiotemporalResamplingPass = createComputePass(kReSTIRGDIShadersFile, "spatiotemporalResampling");

    mFlags.recompileShaders = false;
}

void ReSTIRGDI::prepareResources(RenderContext* pRenderContext)
{
    // Ask for some other refreshes elsewhere to make sure we're all consistent.
    mFlags.clearReservoirs = true;
    mFlags.updateEmissiveLights = true;
    mFlags.updateEmissiveLightsFlux = true;
    mFlags.updateAnalyticLights = true;
    mFlags.updateAnalyticLightsFlux = true;
    mFlags.updateEnvLight = true;

    // Make sure the ReSTIRGDI context has the current screen resolution.
    mRESTIRContextParams.RenderWidth = mFrameDim.x;
    mRESTIRContextParams.RenderHeight = mFrameDim.y;

    // Set the number and size of our presampled tiles.
    mRESTIRContextParams.TileSize = mOptions.presampledTileSize;
    mRESTIRContextParams.TileCount = mOptions.presampledTileCount;
    mRESTIRContextParams.EnvironmentTileSize = mOptions.presampledTileSize;
    mRESTIRContextParams.EnvironmentTileCount = mOptions.presampledTileCount;

    // Create a new ReSTIRGDI context.
    mpRESTIRContext = std::make_unique<restir::Context>(mRESTIRContextParams);

    // Note: Additional resources are allocated lazily in updateLights() and updateEnvMap().

    // Allocate buffer for presampled light tiles (ReSTIRGDI calls this "RIS buffers").
    uint32_t lightTileSampleCount = std::max(mpRESTIRContext->GetRisBufferElementCount(), 1u);
    if (!mpLightTileBuffer || mpLightTileBuffer->getElementCount() < lightTileSampleCount)
    {
        mpLightTileBuffer = mpDevice->createTypedBuffer(ResourceFormat::RG32Uint, lightTileSampleCount);
    }

    // Allocate buffer for compact light info used to improve coherence for presampled light tiles.
    {
        uint32_t elementCount = lightTileSampleCount * 2;
        if (!mpCompactLightInfoBuffer || mpCompactLightInfoBuffer->getElementCount() < elementCount)
        {
            mpCompactLightInfoBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["lightInfo"], elementCount);
        }
    }

    // Allocate buffer for light reservoirs. There are multiple reservoirs (specified by kMaxReservoirs) concatenated together.
    {
        uint32_t elementCount = mpRESTIRContext->GetReservoirBufferElementCount() * kMaxReservoirs;
        if (!mpReservoirBuffer || mpReservoirBuffer->getElementCount() < elementCount)
        {
            mpReservoirBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["reservoirs"], elementCount);
        }
    }

    // Allocate buffer for surface data for current and previous frames.
    {
        uint32_t elementCount = 2 * mFrameDim.x * mFrameDim.y;
        if (!mpSurfaceDataBuffer || mpSurfaceDataBuffer->getElementCount() < elementCount)
        {
            mpSurfaceDataBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["surfaceData"], elementCount);
        }
    }

    // Allocate buffer for neighbor offsets.
    if (!mpNeighborOffsetsBuffer)
    {
        std::vector<uint8_t> offsets(2 * (size_t)mRESTIRContextParams.NeighborOffsetCount);
        mpRESTIRContext->FillNeighborOffsetBuffer(offsets.data());
        mpNeighborOffsetsBuffer = mpDevice->createTypedBuffer(
            ResourceFormat::RG8Snorm,
            mRESTIRContextParams.NeighborOffsetCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            offsets.data()
        );
    }
}

void ReSTIRGDI::setReSTIRGDIFrameParameters()
{
    restir::FrameParameters frameParameters;

    // Set current frame index.
    frameParameters.frameIndex = mFrameIndex;

    // Always enable importance sampling for local lights.
    frameParameters.enableLocalLightImportanceSampling = true;

    // Set the range of local lights.
    frameParameters.firstLocalLight = mLights.getFirstLocalLightIndex();
    frameParameters.numLocalLights = mLights.getLocalLightCount();

    // Set the range of infinite lights.
    frameParameters.firstInfiniteLight = mLights.getFirstInfiniteLightIndex();
    frameParameters.numInfiniteLights = mLights.getInfiniteLightCount();

    // Set the environment light.
    frameParameters.environmentLightPresent = mLights.envLightPresent;
    frameParameters.environmentLightIndex = mLights.getEnvLightIndex();

    // Update the parameters ReSTIRGDI needs when we call its functions in our shaders.
    mpRESTIRContext->FillRuntimeParameters(mRESTIRShaderParams, frameParameters);
}

bool ReSTIRGDI::renderUI(Gui::Widgets& widget)
{
    bool changed = false;

    // Edit a copy of the options and use setOptions() to validate the changes and trigger required
    // actions due to changing them. This unifies the logic independent of using the UI or setOptions() directly.
    Options options = mOptions;

    // Our user-controllable parameters vary depending on if we're doing reuse and what kind
    bool useResampling = (mOptions.mode != Mode::NoResampling);
    bool useTemporalResampling = (mOptions.mode == Mode::TemporalResampling || mOptions.mode == Mode::SpatiotemporalResampling);
    bool useSpatialResampling = (mOptions.mode == Mode::SpatialResampling || mOptions.mode == Mode::SpatiotemporalResampling);

    changed |= widget.dropdown("Mode", options.mode);
    widget.tooltip(
        "Mode.\n\n"
        "NoResampling: No resampling (Talbot RIS from EGSR 2005 \"Importance Resampling for Global Illumination\").\n"
        "SpatialResampling: Spatial resampling only.\n"
        "TemporalResampling: Temporal resampling only.\n"
        "SpatiotemporalResampling: Spatiotemporal resampling."
    );

    if (auto group = widget.group("Light presampling", false))
    {
        changed |= group.var("Tile count", options.presampledTileCount, kMinPresampledTileCount, kMaxPresampledTileCount);
        group.tooltip("Number of precomputed light tiles.");

        changed |= group.var("Tile size", options.presampledTileSize, kMinPresampledTileSize, kMaxPresampledTileSize, 128u);
        group.tooltip("Size of each precomputed light tile (number of samples).");

        changed |= group.checkbox("Store compact light info", options.storeCompactLightInfo);
        group.tooltip("Store compact light info for precomputed light tiles to improve coherence.");
    }

    if (auto group = widget.group("Initial candidate sampling", false))
    {
        changed |= group.var("Local light samples", options.localLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial local light candidate samples.");

        changed |=
            group.var("Infinite light samples", options.infiniteLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial infinite light candidate samples.");

        changed |= group.var("Environment light samples", options.envLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial environment light candidate samples.");

        changed |= group.var("BRDF samples", options.brdfCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial BRDF candidate samples.");

        changed |= group.var("BRDF Cutoff", options.brdfCutoff, 0.f, 1.f);
        group.tooltip("Value in range [0,1] to determine how much to shorten BRDF rays");

        if (useResampling)
        {
            changed |= group.checkbox("Test selected candidate visibility", options.testCandidateVisibility);
            group.tooltip(
                "Test visibility on selected candidate sample before doing resampling.\n\n"
                "Occluded samples have their reseroirs zeroed out, so such a sample never has a chance to contribute "
                "to neighbors. This is especially valuable in multi-room scenes, where occluded lights from a different "
                "room are also unlikely to light neighbors."
            );
        }
    }

    if (useResampling)
    {
        if (auto group = widget.group("Resampling", false))
        {
            changed |= group.dropdown("Bias correction", options.biasCorrection);
            group.tooltip(
                "Bias correction mode.\n\n"
                "Off: Use (1/M) normalization, which is very biased but also very fast.\n"
                "Basic: Use MIS-like normalization but assume that every sample is visible.\n"
                "Pairwise: Use pairwise MIS normalization. Assumes every sample is visibile.\n"
                "RayTraced: Use MIS-like normalization with visibility rays. Unbiased.\n"
            );

            changed |= group.var("Depth threshold", options.depthThreshold, 0.0f, 1.0f, 0.001f);
            group.tooltip("Relative depth difference at which pixels are classified too far apart to be reused (0.1 = 10%).");

            changed |= group.var("Normal threshold", options.normalThreshold, 0.0f, 1.0f, 0.001f);
            group.tooltip("Cosine of the angle between normals, below which pixels are classified too far apart to be reused.");
        }
    }

    if (useSpatialResampling)
    {
        if (auto group = widget.group("Spatial resampling", false))
        {
            changed |= group.var("Sampling radius", options.samplingRadius, kMinSpatialRadius, kMaxSpatialRadius, 0.1f);
            group.tooltip("Screen-space radius for spatial resampling, measured in pixels.");

            changed |= group.var("Sample count", options.spatialSampleCount, kMinSpatialSampleCount, kMaxSpatialSampleCount);
            group.tooltip("Number of neighbor pixels considered for resampling.");

            if (options.mode == Mode::SpatialResampling)
            {
                changed |= group.var("Iterations", options.spatialIterations, kMinSpatialIterations, kMaxSpatialIterations);
                group.tooltip("Number of spatial resampling passes.");
            }
        }
    }

    if (useTemporalResampling)
    {
        if (auto group = widget.group("Temporal resampling", false))
        {
            changed |= group.var("Max history length", options.maxHistoryLength, kMinMaxHistoryLength, kMaxMaxHistoryLength);
            group.tooltip("Maximum history length for temporal reuse, measured in frames.");

            changed |= group.var("Boiling filter strength", options.boilingFilterStrength, 0.0f, 1.0f, 0.001f);
            group.tooltip("0 = off, 1 = full strength.");
        }
    }

    if (auto group = widget.group("Misc"))
    {
        changed |= group.checkbox("Use emissive textures", options.useEmissiveTextures);
        group.tooltip("Use emissive textures to return final sample incident radiance (true is slower and noisier).");

        changed |= group.checkbox("Enable permutation sampling", options.enablePermutationSampling);
        group.tooltip("Enables permuting the pixels sampled from the previous frame (noisier but more denoiser friendly).");
    }

    if (auto group = widget.group("Debugging"))
    {
        mpPixelDebug->renderUI(group);
    }

    if (changed)
        setOptions(options);

    return changed;
}
} // namespace Falcor
