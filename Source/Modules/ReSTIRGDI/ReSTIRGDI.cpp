/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "ReSTIRGDI.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Color/ColorHelpers.slang"

namespace Falcor
{
    namespace
    {
        const char kReflectTypesFile[] = "Modules/ReSTIRGDI/ReflectTypes.cs.slang";
        const char kUpdateEmissiveTriangles[] = "Modules/ReSTIRGDI/UpdateEmissiveTriangles.cs.slang";
        const char kGenerateLightTilesFile[] = "Modules/ReSTIRGDI/GenerateLightTiles.cs.slang";
        const char kInitialResamplingFile[] = "Modules/ReSTIRGDI/InitialResampling.cs.slang";
        const char kTemporalResamplingFile[] = "Modules/ReSTIRGDI/TemporalResampling.cs.slang";
        const char kTemporalResamplingTracePrimaryRays[] = "Modules/ReSTIRGDI/TemporalMSAATracePrimaryRays.cs.slang";
        const char kTemporalResamplingFloatMotionFile[] = "Modules/ReSTIRGDI/TemporalResampling_FloatMotion.cs.slang";
        const char kSpatialResamplingFile[] = "Modules/ReSTIRGDI/SpatialResampling.cs.slang";
        const char kEvaluateFinalSamplesFile[] = "Modules/ReSTIRGDI/EvaluateFinalSamples.cs.slang";

        const std::string kShaderModel = "6_5";
        const uint32_t kNeighborOffsetCount = 8192;
        const uint kColorChannelsPerPixel = 3u;
    }

    ReSTIRGDI::ReSTIRGDI(const ref<Scene>& pScene, const Options& options, const DefineList& ownerDefines)
        : mpScene(pScene)
        , mpDevice(mpScene->getDevice())
        , mOptions(options)
        , mOwnerDefines(ownerDefines)
    {
        FALCOR_ASSERT(mpScene);

        logInfo("ReSTIRGDI() constructor called");

        mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

        // Create compute pass for reflecting data types.
        ProgramDesc desc;
        DefineList defines;
        defines.add(ownerDefines);
        defines.add(mpScene->getSceneDefines());
        defines.add(getLightsDefines());
        defines.add(getDefines());
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines);

        // Create neighbor offset texture.
        mpNeighborOffsets = createNeighborOffsetTexture(kNeighborOffsetCount);
    }

    DefineList ReSTIRGDI::getDefines() const
    {
        DefineList defines;
        defines.add("USE_ALPHA_TEST", mOptions.useAlphaTest ? "1" : "0");
        defines.add("USE_PREV_FRAME_SCENE_DATA", mOptions.usePrevFrameSceneData ? "1" : "0");
        return defines;
    }

    void ReSTIRGDI::setShaderData(const ShaderVar& var) const
    {
        // Set from PathTracer side
        var["surfaceData"] = mpSurfaceData;
        var["normalDepth"] = mpNormalDepthTexture;

        // Used by PathTracer in final shading
        var["finalSamplesForDI"] = mDirectLightingResources.pFinalSamples;
        var["finalPrimaryHitsForDI"] = mDirectLightingResources.pFinalPrimaryHits;
        var["finalSamplesForEmission"] = mEmissiveResources.pFinalSamples;
        var["finalPrimaryHitsForEmission"] = mEmissiveResources.pFinalPrimaryHits;
        var["frameDim"] = mFrameDim;
        var["resampleEmissionMode"] = (uint)mOptions.resampleEmissionMode;
        var["numRestirPasses"] = mOptions.numReSTIRPasses;
    }

    bool ReSTIRGDI::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        dirty |= widget.checkbox("Use M Factor for pairwise MIS", mOptions.useMFactor);

        dirty |= widget.var("Number of ReSTIR passes", mOptions.numReSTIRPasses);

        if (auto group = widget.group("Debugging"))
        {
            mRecompile |= group.dropdown("Debug output", mOptions.debugOutput);
            mpPixelDebug->renderUI(group);

        }

        if (auto group = widget.group("Common options"))
        {
            dirty |= widget.var("Normal threshold", mOptions.normalThreshold, 0.f, 1.f);
            widget.tooltip("Normal cosine threshold for reusing temporal samples or spatial neighbor samples.");

            dirty |= widget.var("Depth threshold", mOptions.depthThreshold, 0.f, 10.f);
            widget.tooltip("Relative depth threshold for reusing temporal samples or spatial neighbor samples.");
        }

        if (auto groupDI = widget.group("ReSTIR DI options", true))
        {
            if (auto group = widget.group("Light selection weights"))
            {
                mRecompile |= group.var("Environment", mOptions.envLightWeight, 0.f, 1.f);
                group.tooltip("Relative weight for selecting the env map when sampling a light.");

                mRecompile |= group.var("Emissive", mOptions.emissiveLightWeight, 0.f, 1.f);
                group.tooltip("Relative weight for selecting an emissive light when sampling a light.");

                mRecompile |= group.var("Analytic", mOptions.analyticLightWeight, 0.f, 1.f);
                group.tooltip("Relative weight for selecting an analytical light when sampling a light.");
            }

            if (auto group = widget.group("Emissive lights"))
            {
                mRecompile |= group.checkbox("Use emissive texture for sampling", mOptions.useEmissiveTextureForSampling);
                group.tooltip("Use emissive texture for light sample evaluation.");

                mRecompile |= group.checkbox("Use emissive texture for shading", mOptions.useEmissiveTextureForShading);
                group.tooltip("Use emissive texture for shading.");

                mRecompile |= group.checkbox("Use local emissive triangles", mOptions.useLocalEmissiveTriangles);
                group.tooltip("Use local emissive triangle data structure (for more efficient sampling/evaluation).");
            }

            if (auto group = widget.group("Light tiles"))
            {
                mRecompile |= group.var("Tile count", mOptions.lightTileCount, 1u, 1024u);
                group.tooltip("Number of light tiles to compute.");

                mRecompile |= group.var("Tile size", mOptions.lightTileSize, 1u, 8192u);
                group.tooltip("Number of lights per light tile.");
            }

            if (auto group = widget.group("Visibility"))
            {
                mRecompile |= group.checkbox("Use alpha test", mOptions.useAlphaTest);
                group.tooltip("Use alpha testing on non-opaque triangles.");

                mRecompile |= group.checkbox("Use initial visibility", mOptions.useInitialVisibility);
                group.tooltip("Check visibility on inital sample.");

                mRecompile |= group.checkbox("Use final visibility", mOptions.useFinalVisibility);
                group.tooltip("Check visibility on final sample.");

                if (mOptions.useFinalVisibility)
                {
                    mRecompile |= group.checkbox("Reuse final visibility", mOptions.reuseFinalVisibility);
                    group.tooltip("Reuse final visibility temporally.");
                }
            }

            if (auto group = widget.group("Initial resampling", true))
            {
                mRecompile |= group.var("Screen tile size", mOptions.screenTileSize, 1u, 128u);
                group.tooltip("Size of screen tile that samples from the same light tile.");

                mRecompile |= group.var("Initial light sample count", mOptions.initialLightSampleCount, 1u, 1024u);
                group.tooltip("Number of initial light samples to resample per pixel.");

                mRecompile |= group.var("Initial BRDF sample count", mOptions.initialBRDFSampleCount, 0u, 16u);
                group.tooltip("Number of initial BRDF samples to resample per pixel.");

                mRecompile |= group.var("Initial Path sample count", mOptions.initialPathSampleCount, 0u, 16u);
                group.tooltip("Number of initial path samples to resample per pixel.");

                dirty |= group.var("BRDF Cutoff", mOptions.brdfCutoff, 0.f, 1.f);
                group.tooltip("Value in range [0,1] to determine how much to shorten BRDF rays.");
            }

            if (auto group = widget.group("Temporal resampling", true))
            {
                mRecompile |= group.checkbox("Use temporal resampling", mOptions.useTemporalResampling);
                mRecompile |= group.checkbox("Use Prev Frame Scene Data", mOptions.usePrevFrameSceneData);
                group.tooltip("Use previous scene BVH, camera, and lights for unbiased resampling in ReSTIR");

                mRecompile |= group.var("Max history length", mOptions.maxHistoryLength, 0u, 100u);
                group.tooltip("Maximum temporal history length.");

                mRecompile |= group.checkbox("Optimize Shift 2RIS", mOptions.optimizeShift2RIS);
                group.tooltip("This will pretrace the temporal resampling rays in advance, and store their resampling data to avoid \n"
                    "tracing redundent rays in one shader call.");

                mRecompile |= group.dropdown("First RIS shift mapping mode", mOptions.temporalShiftMappingModeRIS1);
                group.tooltip("This is the RIS for resampling 4 neighbor pixel's shifting samples in previous frame");

                mRecompile |= group.dropdown("Second RIS shift mapping mode", mOptions.temporalShiftMappingModeRIS2);
                group.tooltip("This is the RIS for resampling the final selected (shifted) sample in previous with current frame's sample");
            }

            if (auto group = widget.group("Spatial resampling", true))
            {
                mRecompile |= group.checkbox("Use spatial resampling", mOptions.useSpatialResampling);

                mRecompile |= group.checkbox("Reject neighbor pixel based on normal & depth", mOptions.rejectNeighborPixelForNormalDepth);
                mRecompile |= group.checkbox("Reject neighbor pixel based on primary hit type", mOptions.rejectNeighborPixelForHitType);

                dirty |= group.var("Iterations", mOptions.spatialIterations, 0u, 8u);
                group.tooltip("Number of spatial resampling iterations.");

                dirty |= group.var("Neighbor count", mOptions.spatialNeighborCount, 0u, 32u);
                group.tooltip("Number of neighbor samples to resample per pixel and iteration.");

                dirty |= group.var("Gather radius", mOptions.spatialGatherRadius, 5u, 40u);
                group.tooltip("Radius to gather samples from.");

                dirty |= group.var("Random replay shift selection weight", mOptions.randomReplaySampleWeight, 0.0f, 1.0f);
                group.tooltip("The constant weight to assign a spatial sample to use lens vertex shift");

                mRecompile |= group.dropdown("How to select sample in MIS shift", mOptions.spatialMisSampleSelection);
                group.tooltip("Different heuristics to choose which shift mapping should be used for each spatial sample if MIS shifts is enabled");

                mRecompile |= group.dropdown("Spatial shift mapping mode", mOptions.spatialShiftMappingMode);
            }

            // ReSTIR common options
            if (auto group = widget.group("ReSTIR options", true))
            {

                mRecompile |= widget.dropdown("Resample emission mode", mOptions.resampleEmissionMode);
                widget.tooltip("How do we handle the emission term in path integration");

                dirty |= widget.checkbox("Scale two shifts weight in MIS mode", mOptions.scaleTwoShiftsWeightForMIS);
                widget.tooltip("Instead of using 0.5/0.5 weight for both shifts in MIS mode, we adaptively compute weights based on CoC.");

                dirty |= widget.checkbox("Better scaling funtion for MIS", mOptions.betterScaleFuntionForMIS);
                widget.tooltip("Instead of using manually tuned piecewise linear function, we use fitted reciprocal function to scale weights.");

                dirty |= widget.var("MIS Scaling function index", mOptions.scalingFunctionIndex, 0u, 5u);
                widget.tooltip("Different fitted MIS weight scaling functions");
            }

            // Other options
            mRecompile |= widget.checkbox("Unbiased", mOptions.unbiased);
            widget.tooltip("Use unbiased version of ReSTIR by querying extra visibility rays.");
        }


        dirty |= mRecompile;

        return dirty;
    }

    bool ReSTIRGDI::onKeyEvents(const KeyboardEvent& keyEvent)
    {
        bool dirty = false;

        return dirty;
    }

    void ReSTIRGDI::setOptions(const Options& options)
    {
        if (std::memcmp(&options, &mOptions, sizeof(Options)) != 0)
        {
            mOptions = options;
            mRecompile = true;
        }
    }

    void ReSTIRGDI::beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint frameCount, float filterRadius, float filterAlpha, float filterNorm)
    {
        mFrameDim = frameDim;
        mFrameIndex = frameCount;
        mOptions.resampleEmissionMode = ResampleEmissionMode::None;

        mFilterRadius = filterRadius;
        mFilterAlpha = filterAlpha;
        mFilterNorm = filterNorm;

        prepareResources(pRenderContext);

        mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
    }

    void ReSTIRGDI::endFrame(RenderContext* pRenderContext)
    {
        // Swap surface data.
        std::swap(mpSurfaceData, mpPrevSurfaceData);

        // Swap reservoirs (and their temporal data)
        std::swap(mDirectLightingResources.pReservoirs, mDirectLightingResources.pPrevReservoirs);
        std::swap(mDirectLightingResources.pResEvalContext, mDirectLightingResources.pPrevResEvalContext);
        std::swap(mDirectLightingResources.pPixelCenterEvalContext, mDirectLightingResources.pPrevPixelCenterEvalContext);
        std::swap(mEmissiveResources.pReservoirs, mEmissiveResources.pPrevReservoirs);
        std::swap(mEmissiveResources.pResEvalContext, mEmissiveResources.pPrevResEvalContext);
        std::swap(mEmissiveResources.pPixelCenterEvalContext, mEmissiveResources.pPrevPixelCenterEvalContext);


        mpPixelDebug->endFrame(pRenderContext);
    }

    void ReSTIRGDI::updateReSTIRDI(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pViewDir)
    {
        FALCOR_PROFILE(pRenderContext, "ReSTIRGDI::updateReSTIRDI");


        prepareLighting(pRenderContext);
        updatePrograms();
        updateEmissiveTriangles(pRenderContext);
        generateLightTiles(pRenderContext);

        auto runResamplingPasses = [&](ResamplingResources& resources)
        {
            initialResampling(pRenderContext, pViewDir, resources);

            temporalResampling(pRenderContext, pMotionVectors, mpTemporalResampling, resources);

            spatialResampling(pRenderContext, resources);
            evaluateFinalSamples(pRenderContext, resources);
        };

        {
            FALCOR_PROFILE(pRenderContext, "ReSTIR DI");
            runResamplingPasses(mDirectLightingResources);
        }
    }

    void ReSTIRGDI::prepareResources(RenderContext* pRenderContext)
    {
        auto reflectVar = mpReflectTypes->getRootVar();

        //logInfo("ReSTIRGDI::prepareResources() called!");

        // Create light tile buffers.
        {
            uint32_t elementCount = mOptions.lightTileCount * mOptions.lightTileSize;
            if (!mpLightTileData || mpLightTileData->getElementCount() < elementCount)
            {
                mpLightTileData = mpDevice->createStructuredBuffer(reflectVar["lightTileData"], elementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
        }

        // Create global buffers (stay unchanged in resampling)
        {
            // Create screen sized buffers.
            uint32_t elementCount = mFrameDim.x * mFrameDim.y;
            if (!mpSurfaceData || mpSurfaceData->getElementCount() < elementCount)
            {
                mpSurfaceData = mpDevice->createStructuredBuffer(reflectVar["surfaceData"], elementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!mpPrevSurfaceData || mpPrevSurfaceData->getElementCount() < elementCount)
            {
                mpPrevSurfaceData = mpDevice->createStructuredBuffer(reflectVar["surfaceData"], elementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }


            // Create normal/depth texture.
            if (!mpNormalDepthTexture || mpNormalDepthTexture->getWidth() != mFrameDim.x || mpNormalDepthTexture->getHeight() != mFrameDim.y)
            {
                mpNormalDepthTexture = mpDevice->createTexture2D(mFrameDim.x, mFrameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }
            if (!mpPrevNormalDepthTexture || mpPrevNormalDepthTexture->getWidth() != mFrameDim.x || mpPrevNormalDepthTexture->getHeight() != mFrameDim.y)
            {
                mpPrevNormalDepthTexture = mpDevice->createTexture2D(mFrameDim.x, mFrameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
            }

            // Create debug texture.
            if (!mpDebugOutputTexture || mpDebugOutputTexture->getWidth() != mFrameDim.x || mpDebugOutputTexture->getHeight() != mFrameDim.y)
            {
                mpDebugOutputTexture = mpDevice->createTexture2D(mFrameDim.x, mFrameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }
        }

        // Create resampling buffers
        auto initResamplingResources = [&](ResamplingResources& resources)
        {
            // How many primary rays we need to pe-trace in MSAA pass
            resources.perPixelMsaaShiftsCount = mOptions.temporalShiftMappingModeRIS1 == ShiftMappingModeInReusing::MIS ? 16u : 8u;

            // Create screen sized buffers.
            uint32_t elementCount = mFrameDim.x * mFrameDim.y;
            uint reservoirsCount = elementCount * mOptions.numReSTIRPasses;
            if (!resources.pReservoirs || resources.pReservoirs->getElementCount() < reservoirsCount)
            {
                resources.pReservoirs = mpDevice->createStructuredBuffer(reflectVar["reservoirs"], reservoirsCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pPrevReservoirs || resources.pPrevReservoirs->getElementCount() < reservoirsCount)
            {
                resources.pPrevReservoirs = mpDevice->createStructuredBuffer(reflectVar["reservoirs"], reservoirsCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pResEvalContext || resources.pResEvalContext->getElementCount() < reservoirsCount)
            {
                resources.pResEvalContext = mpDevice->createStructuredBuffer(reflectVar["resEvalContext"], reservoirsCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pPrevResEvalContext || resources.pPrevResEvalContext->getElementCount() < reservoirsCount)
            {
                resources.pPrevResEvalContext = mpDevice->createStructuredBuffer(reflectVar["resEvalContext"], reservoirsCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pPixelCenterEvalContext || resources.pPixelCenterEvalContext->getElementCount() < elementCount)
            {
                resources.pPixelCenterEvalContext = mpDevice->createStructuredBuffer(reflectVar["resEvalContext"], elementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pPrevPixelCenterEvalContext || resources.pPrevPixelCenterEvalContext->getElementCount() < elementCount)
            {
                resources.pPrevPixelCenterEvalContext = mpDevice->createStructuredBuffer(reflectVar["resEvalContext"], elementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pFinalSamples || resources.pFinalSamples->getElementCount() < reservoirsCount)
            {
                resources.pFinalSamples = mpDevice->createStructuredBuffer(reflectVar["finalSamples"], reservoirsCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            if (!resources.pFinalPrimaryHits || resources.pFinalPrimaryHits->getElementCount() < reservoirsCount)
            {
                resources.pFinalPrimaryHits = mpDevice->createStructuredBuffer(reflectVar["finalPrimaryHits"], reservoirsCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
        };

        // Initialize resampling resources
        initResamplingResources(mDirectLightingResources);
        mDirectLightingResources.type = ResamplingResourceType::DirectLighting;
        initResamplingResources(mEmissiveResources);
        mEmissiveResources.type = ResamplingResourceType::Emission;
    }

    void ReSTIRGDI::prepareLighting(RenderContext* pRenderContext)
    {
        if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged)) mRecompile = true;

        // Setup alias table for env light.
        if (mpScene->useEnvLight())
        {
            const auto& envMap = mpScene->getEnvMap();
            if (!mpEnvLightLuminance || !mpEnvLightAliasTable)
            {
                const auto& texture = mpScene->getEnvMap()->getEnvMap();
                std::vector<float3> radiances;
                auto luminances = computeEnvLightLuminance(pRenderContext, texture, radiances);
                mpEnvLightLuminance = mpDevice->createTypedBuffer<float>((uint32_t)luminances.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, luminances.data());
                mpEnvLightAliasTable = buildEnvLightAliasTable(texture->getWidth(), texture->getHeight(), luminances, mRng);
                mRecompile = true;
            }

            mEnvLightLuminanceFactor = luminance(envMap->getIntensity() * envMap->getTint());
        }
        else
        {
            if (mpEnvLightLuminance)
            {
                mpEnvLightLuminance = nullptr;
                mpEnvLightAliasTable = nullptr;
                mRecompile = true;
            }
        }

        // Setup alias table for emissive lights.
        if (mpScene->getRenderSettings().useEmissiveLights)
        {
            if (!mpEmissiveLightAliasTable)
            {
                auto lightCollection = mpScene->getLightCollection(pRenderContext);
                lightCollection->prepareSyncCPUData(pRenderContext);
                lightCollection->update(pRenderContext);
                if (lightCollection->getActiveLightCount(pRenderContext) > 0)
                {
                    mpEmissiveTriangles = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["emissiveTriangles"],
                        lightCollection->getTotalLightCount(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                        MemoryType::DeviceLocal, nullptr, false);
                    mpEmissiveLightAliasTable = buildEmissiveLightAliasTable(pRenderContext, lightCollection, mRng);
                    mRecompile = true;
                }
            }
        }
        else
        {
            if (mpEmissiveTriangles)
            {
                mpEmissiveTriangles = nullptr;
                mpEmissiveLightAliasTable = nullptr;
                mRecompile = true;
            }
        }

        // Setup alias table for analytic lights.
        if (mpScene->useAnalyticLights())
        {
            if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::LightCountChanged)) mpAnalyticLightAliasTable = nullptr;
            if (!mpAnalyticLightAliasTable)
            {
                std::vector<ref<Light>> lights;
                for (uint32_t i = 0; i < mpScene->getLightCount(); ++i)
                {
                    const auto& light = mpScene->getLight(i);
                    if (light->isActive()) lights.push_back(mpScene->getLight(i));
                }
                if (!lights.empty())
                {
                    mpAnalyticLightAliasTable = buildAnalyticLightAliasTable(pRenderContext, lights, mRng);
                    mRecompile = true;
                }
            }
        }
        else
        {
            if (mpAnalyticLightAliasTable)
            {
                mpAnalyticLightAliasTable = nullptr;
                mRecompile = true;
            }
        }

        // Compute light selection probabilities.
        auto& probs = mLightSelectionProbabilities;
        probs.envLight = mpEnvLightAliasTable ? mOptions.envLightWeight : 0.f;
        probs.emissiveLights = mpEmissiveLightAliasTable ? mOptions.emissiveLightWeight : 0.f;
        probs.analyticLights = mpAnalyticLightAliasTable ? mOptions.analyticLightWeight : 0.f;
        float total = probs.envLight + probs.emissiveLights + probs.analyticLights;
        if (total > 0.f)
        {
            probs.envLight /= total;
            probs.emissiveLights /= total;
            probs.analyticLights /= total;
        }
    }

    void ReSTIRGDI::updatePrograms()
    {
        if (!mRecompile) return;

        DefineList commonDefines;

        commonDefines.add(mOwnerDefines);
        commonDefines.add(mpScene->getSceneDefines());
        commonDefines.add(getLightsDefines());
        commonDefines.add(getDefines());
        commonDefines.add("DEBUG_OUTPUT", std::to_string((uint32_t)mOptions.debugOutput));

        auto shaderModules = mpScene->getShaderModules();
        auto typeConformances = mpScene->getTypeConformances();

        // UpdateEmissiveTriangles
        {
            DefineList defines = commonDefines;

            if (!mpUpdateEmissiveTriangles)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kUpdateEmissiveTriangles).csEntry("main");
                mpUpdateEmissiveTriangles = ComputePass::create(mpDevice, desc, defines, false);
            }

            mpUpdateEmissiveTriangles->getProgram()->addDefines(defines);
            mpUpdateEmissiveTriangles->setVars(nullptr);
        }

        // GenerateLightTiles
        {
            DefineList defines = commonDefines;

            defines.add("LIGHT_TILE_COUNT", std::to_string(mOptions.lightTileCount));
            defines.add("LIGHT_TILE_SIZE", std::to_string(mOptions.lightTileSize));

            auto [envLightSampleCount, emissiveLightSampleCount, analyticLightSampleCount] = mLightSelectionProbabilities.getSampleCount(mOptions.lightTileSize);
            defines.add("ENV_LIGHT_SAMPLE_COUNT", std::to_string(envLightSampleCount));
            defines.add("EMISSIVE_LIGHT_SAMPLE_COUNT", std::to_string(emissiveLightSampleCount));
            defines.add("ANALYTIC_LIGHT_SAMPLE_COUNT", std::to_string(analyticLightSampleCount));

            if (!mpGenerateLightTiles)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kGenerateLightTilesFile).csEntry("main");
                mpGenerateLightTiles = ComputePass::create(mpDevice, desc, defines, false);
            }

            mpGenerateLightTiles->getProgram()->addDefines(defines);
            mpGenerateLightTiles->setVars(nullptr);
        }

        // InitialResampling
        {
            DefineList defines = commonDefines;

            defines.add("LIGHT_TILE_COUNT", std::to_string(mOptions.lightTileCount));
            defines.add("LIGHT_TILE_SIZE", std::to_string(mOptions.lightTileSize));

            defines.add("SCREEN_TILE_SIZE", std::to_string(mOptions.screenTileSize));
            defines.add("INITIAL_LIGHT_SAMPLE_COUNT", std::to_string(mOptions.initialLightSampleCount));
            defines.add("INITIAL_BRDF_SAMPLE_COUNT", std::to_string(mOptions.initialBRDFSampleCount));
            defines.add("INITIAL_PATH_SAMPLE_COUNT", std::to_string(mOptions.initialPathSampleCount));

            // Only need to check visibility if either temporal or spatial reuse is active.
            bool checkVisibility = mOptions.useInitialVisibility & (mOptions.useTemporalResampling || mOptions.useSpatialResampling);
            defines.add("CHECK_VISIBILITY", checkVisibility ? "1" : "0");

            if (!mpInitialResampling)
            {
                ProgramDesc desc;
                desc.addShaderModules(shaderModules);
                desc.addShaderLibrary(kInitialResamplingFile).csEntry("main");
                desc.addTypeConformances(typeConformances);
                mpInitialResampling = ComputePass::create(mpDevice, desc, defines, false);

                logInfo("Reload mpInitialResampling shader");
            }

            mpInitialResampling->getProgram()->addDefines(defines);
            mpInitialResampling->setVars(nullptr);
        }

        // TemporalResampling
        {
            DefineList defines = commonDefines;

            defines.add("MAX_HISTORY_LENGTH", std::to_string(mOptions.maxHistoryLength));
            defines.add("UNBIASED", mOptions.unbiased ? "1" : "0");

            if (!mpTemporalResampling)
            {
                ProgramDesc desc;
                desc.addShaderModules(shaderModules);
                desc.addShaderLibrary(kTemporalResamplingFile).csEntry("main");
                desc.addTypeConformances(typeConformances);
                mpTemporalResampling = ComputePass::create(mpDevice, desc, defines, false);

                logInfo("Reload mpTemporalResampling shader");
            }

            mpTemporalResampling->getProgram()->addDefines(defines);
            mpTemporalResampling->setVars(nullptr);
        }

        // Spatial Resampling
        {
            DefineList defines = commonDefines;

            defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(mpNeighborOffsets->getWidth()));
            defines.add("UNBIASED", mOptions.unbiased ? "1" : "0");

            if (!mpSpatialResampling)
            {
                ProgramDesc desc;
                desc.addShaderModules(shaderModules);
                desc.addShaderLibrary(kSpatialResamplingFile).csEntry("main");
                desc.addTypeConformances(typeConformances);
                mpSpatialResampling = ComputePass::create(mpDevice, desc, defines, false);
            }

            mpSpatialResampling->getProgram()->addDefines(defines);
            mpSpatialResampling->setVars(nullptr);
        }

        // EvaluateFinalSamples
        {
            DefineList defines = commonDefines;

            defines.add("UNBIASED", mOptions.unbiased ? "1" : "0");
            defines.add("USE_VISIBILITY", mOptions.useFinalVisibility ? "1" : "0");
            defines.add("REUSE_VISIBILITY", (mOptions.useFinalVisibility && mOptions.reuseFinalVisibility) ? "1" : "0");
            defines.add("MAX_HISTORY_LENGTH", std::to_string(mOptions.maxHistoryLength));


            if (!mpEvaluateFinalSamples)
            {
                ProgramDesc desc;
                desc.addShaderModules(shaderModules);
                desc.addShaderLibrary(kEvaluateFinalSamplesFile).csEntry("main");
                desc.addTypeConformances(typeConformances);
                mpEvaluateFinalSamples = ComputePass::create(mpDevice, desc, defines, false);
            }

            mpEvaluateFinalSamples->getProgram()->addDefines(defines);
            mpEvaluateFinalSamples->setVars(nullptr);
        }

        mRecompile = false;
        mResetTemporalReservoirs = true;
    }

    void ReSTIRGDI::updateEmissiveTriangles(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE(pRenderContext, "updateEmissiveTriangles");

        if (!mOptions.useLocalEmissiveTriangles || !mpEmissiveTriangles) return;

        mpScene->bindShaderData(mpUpdateEmissiveTriangles->getRootVar()["gScene"]);

        auto var = mpUpdateEmissiveTriangles->getRootVar()["CB"]["gUpdateEmissiveTriangles"];

        uint32_t triangleCount = mpEmissiveTriangles->getElementCount();

        var["emissiveTriangles"] = mpEmissiveTriangles;
        var["emissiveTriangleCount"] = triangleCount;

        mpUpdateEmissiveTriangles->execute(pRenderContext, uint3(triangleCount, 1, 1));
    }

    void ReSTIRGDI::generateLightTiles(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE(pRenderContext, "generateLightTiles");

        mpScene->bindShaderData(mpGenerateLightTiles->getRootVar()["gScene"]);

        auto var = mpGenerateLightTiles->getRootVar()["CB"]["gGenerateLightTiles"];

        var["lightTileData"] = mpLightTileData;
        setLightsShaderData(var["lights"]);
        var["frameIndex"] = mFrameIndex;

        mpGenerateLightTiles->execute(pRenderContext, uint3(mOptions.lightTileSize, mOptions.lightTileCount, 1));
    }

    void ReSTIRGDI::initialResampling(RenderContext* pRenderContext, const ref<Texture>& pViewDir, ResamplingResources& resources)
    {
        FALCOR_PROFILE(pRenderContext, "initialResampling");

        mpScene->bindShaderData(mpInitialResampling->getRootVar()["gScene"]);

        auto rootVar = mpInitialResampling->getRootVar();
        rootVar["CB"]["resampleResourceType"] = uint(resources.type);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(mpInitialResampling->getProgram(), rootVar);

        auto var = rootVar["CB"]["gInitialResampling"];
        var["surfaceData"] = mpSurfaceData;
        var["normalDepth"] = mpNormalDepthTexture;
        var["viewDir"] = pViewDir;
        var["lightTileData"] = mpLightTileData;
        var["reservoirs"] = resources.pReservoirs;
        var["resEvalContext"] = resources.pResEvalContext;
        var["pixelCenterEvalContext"] = resources.pPixelCenterEvalContext;
        var["debugOutput"] = mpDebugOutputTexture;
        var["frameDim"] = mFrameDim;
        var["frameIndex"] = mFrameIndex;
        var["brdfCutoff"] = mOptions.brdfCutoff;
        var["resampleEmissionMode"] = (uint)mOptions.resampleEmissionMode;

        var["filterRadius"] = mFilterRadius;
        var["filterAlpha"] = mFilterAlpha;
        var["filterNorm"] = mFilterNorm;

        setLightsShaderData(var["lights"]);
        setResamplingShaderData(rootVar["SharedResamplingCB"]);

        for (uint i = 0; i < mOptions.numReSTIRPasses; i++)
        {
            var["restirPassIdx"] = i;
            mpInitialResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }
    }

    
    void ReSTIRGDI::temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<ComputePass>& pTemporalPass,
        ResamplingResources& resources)
    {
        FALCOR_PROFILE(pRenderContext, "temporalResampling");

        if (mResetTemporalReservoirs)
        {
            mResetTemporalReservoirs = false;
            return;
        }

        if (!mOptions.useTemporalResampling) return;

        mpScene->bindShaderData(pTemporalPass->getRootVar()["gScene"]);

        auto rootVar = pTemporalPass->getRootVar();
        rootVar["CB"]["resampleResourceType"] = uint(resources.type);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pTemporalPass->getProgram(), rootVar);

        auto var = rootVar["CB"]["gTemporalResampling"];
        var["motionVectors"] = pMotionVectors;
        var["reservoirs"] = resources.pReservoirs;
        var["prevReservoirs"] = resources.pPrevReservoirs;
        var["resEvalContext"] = resources.pResEvalContext;
        var["prevResEvalContext"] = resources.pPrevResEvalContext;
        var["frameDim"] = mFrameDim;
        var["frameIndex"] = mFrameIndex;
        var["useMFactor"] = mOptions.useMFactor;

        setResamplingShaderData(rootVar["SharedResamplingCB"]);

        if (resources.type == ResamplingResourceType::DirectLighting)
        {
            var["debugOutput"] = mpDebugOutputTexture;
            setLightsShaderData(var["lights"]);
        }

        for (uint i = 0; i < mOptions.numReSTIRPasses; i++)
        {
            var["restirPassIdx"] = i;
            pTemporalPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }
    }

    void ReSTIRGDI::spatialResampling(RenderContext* pRenderContext, ResamplingResources& resources)
    {
        FALCOR_PROFILE(pRenderContext, "spatialResampling");

        if (!mOptions.useSpatialResampling) return;

        mpScene->bindShaderData(mpSpatialResampling->getRootVar()["gScene"]);

        auto rootVar = mpSpatialResampling->getRootVar();
        rootVar["CB"]["resampleResourceType"] = uint(resources.type);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(mpSpatialResampling->getProgram(), rootVar);

        auto var = rootVar["CB"]["gSpatialResampling"];
        var["pixelCenterEvalContext"] = resources.pPixelCenterEvalContext;
        var["normalDepth"] = mpNormalDepthTexture;
        var["debugOutput"] = mpDebugOutputTexture;
        var["neighborOffsets"] = mpNeighborOffsets;
        var["frameDim"] = mFrameDim;
        var["frameIndex"] = mFrameIndex;
        var["normalThreshold"] = mOptions.normalThreshold;
        var["depthThreshold"] = mOptions.depthThreshold;
        var["neighborCount"] = mOptions.spatialNeighborCount;
        var["gatherRadius"] = (float)mOptions.spatialGatherRadius;
        var["useMFactor"] = mOptions.useMFactor;
        var["shiftMappingMode"] = (uint)mOptions.spatialShiftMappingMode;
        var["randomReplaySampleWeight"] = mOptions.randomReplaySampleWeight;
        var["misSampleSelection"] = (uint)mOptions.spatialMisSampleSelection;
        var["rejectNeighborPixelForNormalDepth"] = mOptions.rejectNeighborPixelForNormalDepth;
        var["rejectNeighborPixelForHitType"] = mOptions.rejectNeighborPixelForHitType;
        var["resampleEmissionMode"] = (uint)mOptions.resampleEmissionMode;
        setLightsShaderData(var["lights"]);
        setResamplingShaderData(rootVar["SharedResamplingCB"]);

        for (uint32_t iteration = 0; iteration < mOptions.spatialIterations; ++iteration)
        {
            std::swap(resources.pReservoirs, resources.pPrevReservoirs);
            std::swap(resources.pResEvalContext, resources.pPrevResEvalContext);
            var["reservoirs"] = resources.pReservoirs;
            var["prevReservoirs"] = resources.pPrevReservoirs;
            var["resEvalContext"] = resources.pResEvalContext;
            var["prevResEvalContext"] = resources.pPrevResEvalContext;
            var["spatialPassIdx"] = iteration;

            for (uint i = 0; i < mOptions.numReSTIRPasses; i++)
            {
                var["restirPassIdx"] = i;
                mpSpatialResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
            }
        }
    }

    void ReSTIRGDI::evaluateFinalSamples(RenderContext* pRenderContext, ResamplingResources& resources)
    {
        FALCOR_PROFILE(pRenderContext, "evaluateFinalSamples");

        mpScene->bindShaderData(mpEvaluateFinalSamples->getRootVar()["gScene"]);

        auto rootVar = mpEvaluateFinalSamples->getRootVar();
        rootVar["CB"]["resampleResourceType"] = uint(resources.type);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(mpEvaluateFinalSamples->getProgram(), rootVar);

        auto var = rootVar["CB"]["gEvaluateFinalSamples"];
        var["reservoirs"] = resources.pReservoirs;
        var["resEvalContext"] = resources.pResEvalContext;
        var["finalSamples"] = resources.pFinalSamples;
        var["finalPrimaryHits"] = resources.pFinalPrimaryHits;
        var["debugOutput"] = mpDebugOutputTexture;
        var["frameDim"] = mFrameDim;
        var["frameIndex"] = mFrameIndex;
        var["numRestirPasses"] = mOptions.numReSTIRPasses;
        setLightsShaderData(var["lights"]);
        setResamplingShaderData(rootVar["SharedResamplingCB"]);

        mpEvaluateFinalSamples->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    DefineList ReSTIRGDI::getLightsDefines() const
    {
        DefineList defines;

        uint32_t envIndexBits = 26;
        uint32_t envPositionBits = 4;

        uint32_t emissiveIndexBits = 22;
        uint32_t emissivePositionBits = 8;

        uint32_t analyticIndexBits = 14;
        uint32_t analyticPositionBits = 16;

        auto computeIndexPositionBits = [] (const std::unique_ptr<AliasTable>& aliasTable, uint32_t& indexBits, uint32_t& positionBits)
        {
            if (!aliasTable) return;
            uint32_t count = aliasTable->getCount();
            indexBits = 0;
            while (count > 0) {
                ++indexBits;
                count >>= 1;
            }
            if (indexBits & 1) ++indexBits;
            if (indexBits >= 30) throw RuntimeError("Count too large to be represented in 30 bits");
            positionBits = 30 - indexBits;
        };

        computeIndexPositionBits(mpEnvLightAliasTable, envIndexBits, envPositionBits);
        computeIndexPositionBits(mpEmissiveLightAliasTable, emissiveIndexBits, emissivePositionBits);
        computeIndexPositionBits(mpAnalyticLightAliasTable, analyticIndexBits, analyticPositionBits);

        defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
        defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");

        defines.add("LIGHT_SAMPLE_ENV_INDEX_BITS", std::to_string(envIndexBits));
        defines.add("LIGHT_SAMPLE_ENV_POSITION_BITS", std::to_string(envPositionBits));
        defines.add("LIGHT_SAMPLE_EMISSIVE_INDEX_BITS", std::to_string(emissiveIndexBits));
        defines.add("LIGHT_SAMPLE_EMISSIVE_POSITION_BITS", std::to_string(emissivePositionBits));
        defines.add("LIGHT_SAMPLE_ANALYTIC_INDEX_BITS", std::to_string(analyticIndexBits));
        defines.add("LIGHT_SAMPLE_ANALYTIC_POSITION_BITS", std::to_string(analyticPositionBits));

        defines.add("USE_EMISSIVE_TEXTURE_FOR_SAMPLING", mOptions.useEmissiveTextureForSampling ? "1" : "0");
        defines.add("USE_EMISSIVE_TEXTURE_FOR_SHADING", mOptions.useEmissiveTextureForShading ? "1" : "0");
        defines.add("USE_LOCAL_EMISSIVE_TRIANGLES", mOptions.useLocalEmissiveTriangles ? "1" : "0");

        return defines;
    }

    void ReSTIRGDI::setLightsShaderData(const ShaderVar& var) const
    {
        var["envLightLuminance"] = mpEnvLightLuminance;
        var["emissiveTriangles"] = mpEmissiveTriangles;

        if (mpEnvLightAliasTable) mpEnvLightAliasTable->bindShaderData(var["envLightAliasTable"]);
        if (mpEmissiveLightAliasTable) mpEmissiveLightAliasTable->bindShaderData(var["emissiveLightAliasTable"]);
        if (mpAnalyticLightAliasTable) mpAnalyticLightAliasTable->bindShaderData(var["analyticLightAliasTable"]);

        var["envLightLuminanceFactor"] = mEnvLightLuminanceFactor;

        var["envLightSelectionProbability"] = mLightSelectionProbabilities.envLight;
        var["emissiveLightSelectionProbability"] = mLightSelectionProbabilities.emissiveLights;
        var["analyticLightSelectionProbability"] = mLightSelectionProbabilities.analyticLights;
    }


    void ReSTIRGDI::setResamplingShaderData(const ShaderVar& var) const
    {
        var["scaleTwoShiftsWeightForMIS"] = mOptions.scaleTwoShiftsWeightForMIS;
        var["betterScaleFuntionForMIS"] = mOptions.betterScaleFuntionForMIS;
        var["scalingFunctionIndex"] = mOptions.scalingFunctionIndex;
    }

    std::vector<float> ReSTIRGDI::computeEnvLightLuminance(RenderContext* pRenderContext, const ref<Texture>& texture, std::vector<float3>& radiances)
    {
        FALCOR_ASSERT(texture);

        uint32_t width = texture->getWidth();
        uint32_t height = texture->getHeight();

        // Read texel data from the env map texture so we can create an alias table of samples proportional to intensity.
        std::vector<uint8_t> texelsRaw;
        if (getFormatType(texture->getFormat()) == FormatType::Float)
        {
            texelsRaw = pRenderContext->readTextureSubresource(texture.get(), 0);
        }
        else
        {
            auto floatTexture = mpDevice->createTexture2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
            pRenderContext->blit(texture->getSRV(), floatTexture->getRTV());
            texelsRaw = pRenderContext->readTextureSubresource(floatTexture.get(), 0);
        }

        uint32_t texelCount = width * height;
        uint32_t channelCount = getFormatChannelCount(texture->getFormat());
        const float* texels = reinterpret_cast<const float*>(texelsRaw.data());

        std::vector<float> luminances(texelCount);
        radiances.resize(texelCount);

        if (channelCount == 1)
        {
            for (uint32_t i = 0; i < texelCount; ++i)
            {
                luminances[i] = texels[0];
                texels += channelCount;
            }
        }
        else if (channelCount == 3 || channelCount == 4)
        {
            for (uint32_t i = 0; i < texelCount; ++i)
            {
                luminances[i] = luminance(float3(texels[0], texels[1], texels[2]));
                radiances[i] = float3(texels[0], texels[1], texels[2]);
                texels += channelCount;
            }
        }
        else
        {
            throw RuntimeError("Invalid number of channels in env map");
        }

        return luminances;
    }

    std::unique_ptr<AliasTable> ReSTIRGDI::buildEnvLightAliasTable(uint32_t width, uint32_t height, const std::vector<float>& luminances, std::mt19937& rng)
    {
        FALCOR_ASSERT(luminances.size() == width * height);

        std::vector<float> weights(width * height);

        // Computes weights as luminance multiplied by the texel's solid angle.
        for (uint32_t i = 0, y = 0; y < height; ++y)
        {
            float theta = (float)M_PI * (y + 0.5f) / height;
            float solidAngle = (2.f * (float)M_PI / width) * ((float)M_PI / height) * std::sin(theta);

            for (uint32_t x = 0; x < width; ++x, ++i)
            {
                weights[i] = luminances[i] * solidAngle;
            }
        }

        return std::make_unique<AliasTable>(mpDevice, std::move(weights), rng);
    }

    std::unique_ptr<AliasTable> ReSTIRGDI::buildEmissiveLightAliasTable(RenderContext* pRenderContext, const ref<LightCollection>& lightCollection, std::mt19937& rng)
    {
        FALCOR_ASSERT(lightCollection);

        lightCollection->update(pRenderContext);

        const auto& triangles = lightCollection->getMeshLightTriangles(pRenderContext);

        std::vector<float> weights(triangles.size());

        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] = luminance(triangles[i].averageRadiance) * triangles[i].area;
        }

        return std::make_unique<AliasTable>(mpDevice, std::move(weights), rng);
    }

    std::unique_ptr<AliasTable> ReSTIRGDI::buildAnalyticLightAliasTable(RenderContext* pRenderContext, const std::vector<ref<Light>>& lights, std::mt19937& rng)
    {
        std::vector<float> weights(lights.size());

        for (size_t i = 0; i < weights.size(); ++i)
        {
            // TODO: Use weight based on light power.
            weights[i] = 1.f;
        }

        return std::make_unique<AliasTable>(mpDevice, std::move(weights), rng);
    }

    ref<Texture> ReSTIRGDI::createNeighborOffsetTexture(uint32_t sampleCount)
    {
        std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
        const int R = 254;
        const float phi2 = 1.f / 1.3247179572447f;
        float u = 0.5f;
        float v = 0.5f;
        for (uint32_t index = 0; index < sampleCount * 2;)
        {
            u += phi2;
            v += phi2 * phi2;
            if (u >= 1.f) u -= 1.f;
            if (v >= 1.f) v -= 1.f;

            float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
            if (rSq > 0.25f) continue;

            offsets[index++] = int8_t((u - 0.5f) * R);
            offsets[index++] = int8_t((v - 0.5f) * R);
        }

        return mpDevice->createTexture1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
    }

    void ReSTIRGDI::setOwnerDefines(DefineList defines)
    {
        mOwnerDefines = defines;
    }
}
