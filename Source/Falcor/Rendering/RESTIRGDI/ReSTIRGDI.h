#pragma once
#include "Core/Macros.h"
#include "Core/Enum.h"
#include "Utils/Properties.h"
#include "Utils/Debug/PixelDebug.h"
#include "Scene/Scene.h"
#include <memory>
#include <type_traits>
#include <vector>
#include <stdint.h>
#include "RestirgdiParameters.h"

namespace restir
{
struct uint3
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct float3
{
    float x;
    float y;
    float z;
};

// Checkerboard sampling modes match those used in NRD, based on frameIndex:
// Even frame(0)  Odd frame(1)   ...
//     B W             W B
//     W B             B W
// BLACK and WHITE modes define cells with VALID data
enum class CheckerboardMode : uint32_t
{
    Off = 0,
    Black = 1,
    White = 2
};

struct ContextParameters
{
    uint32_t TileSize = 1024;
    uint32_t TileCount = 128;
    uint32_t NeighborOffsetCount = 8192;
    uint32_t RenderWidth = 0;
    uint32_t RenderHeight = 0;
    uint32_t EnvironmentTileSize = 1024;
    uint32_t EnvironmentTileCount = 128;

    CheckerboardMode CheckerboardSamplingMode = CheckerboardMode::Off;
};

struct FrameParameters
{
    // Linear index of the current frame, used to determine the checkerboard field.
    uint32_t frameIndex = 0;

    // Index of the first local light in the light buffer.
    uint32_t firstLocalLight = 0;

    // Number of local lights available on this frame.
    uint32_t numLocalLights = 0;

    // Index of the first infinite light in the light buffer.
    uint32_t firstInfiniteLight = 0;

    // Number of infinite lights available on this frame. They must be indexed
    // immediately following the local lights.
    uint32_t numInfiniteLights = 0;

    // Enables the use of an importance sampled environment map light.
    bool environmentLightPresent = false;

    // Index of the importance environment light in the light buffer.
    uint32_t environmentLightIndex = RESTIR_INVALID_LIGHT_INDEX;

    // Use image-based importance sampling for local lights
    bool enableLocalLightImportanceSampling = false;
};

class Context
{
private:
    ContextParameters m_Params;

    uint32_t m_ReservoirBlockRowPitch = 0;
    uint32_t m_ReservoirArrayPitch = 0;

    uint32_t m_RegirCellOffset = 0;
    uint32_t m_OnionCells = 0;
    std::vector<RESTIR_OnionLayerGroup> m_OnionLayers;
    std::vector<RESTIR_OnionRing> m_OnionRings;
    float m_OnionCubicRootFactor = 0.f;
    float m_OnionLinearFactor = 0.f;

    void ComputeOnionJitterCurve();

public:
    Context(const ContextParameters& params);
    const ContextParameters& GetParameters() const;

    uint32_t GetRisBufferElementCount() const;
    uint32_t GetReservoirBufferElementCount() const;

    void FillRuntimeParameters(RESTIR_ResamplingRuntimeParameters& runtimeParams, const FrameParameters& frame) const;

    void FillNeighborOffsetBuffer(uint8_t* buffer) const;

};

void ComputePdfTextureSize(uint32_t maxItems, uint32_t& outWidth, uint32_t& outHeight, uint32_t& outMipLevels);

} // namespace restir

namespace Falcor
{
class RenderContext;
class FALCOR_API ReSTIRGDI
{
public:
    /** ReSTIRGDI sampling modes.
     */
    enum class Mode
    {
        NoResampling = 1,             ///< No resampling.
        SpatialResampling = 2,        ///< Spatial resampling only.
        TemporalResampling = 3,       ///< Temporal resampling only.
        SpatiotemporalResampling = 4, ///< Spatiotemporal resampling.
    };

    FALCOR_ENUM_INFO(
        Mode,
        {
            {Mode::NoResampling, "NoResampling"},
            {Mode::SpatialResampling, "SpatialResampling"},
            {Mode::TemporalResampling, "TemporalResampling"},
            {Mode::SpatiotemporalResampling, "SpatiotemporalResampling"},
        }
    );

    /** Bias correction modes.
     */
    enum class BiasCorrection
    {
        Off = 0,       ///< Use (1/M) normalization, which is very biased but also very fast.
        Basic = 1,     ///< Use MIS-like normalization but assume that every sample is visible.
        Pairwise = 2,  ///< Use pairwise MIS normalization. Assumes every sample is visible.
        RayTraced = 3, ///< Use MIS-like normalization with visibility rays. Unbiased.
    };

    FALCOR_ENUM_INFO(
        BiasCorrection,
        {
            {BiasCorrection::Off, "Off"},
            {BiasCorrection::Basic, "Basic"},
            {BiasCorrection::Pairwise, "Pairwise"},
            {BiasCorrection::RayTraced, "RayTraced"},
        }
    );

    /** Configuration options, with generally reasonable defaults.
     */
    struct Options
    {
        Mode mode = Mode::SpatiotemporalResampling; ///< ReSTIRGDI sampling mode.

        // Light presampling options.
        uint32_t presampledTileCount = 128; ///< Number of precomputed light tiles.
        uint32_t presampledTileSize = 1024; ///< Size of each precomputed light tile (number of samples).
        bool storeCompactLightInfo = true;  ///< Store compact light info for precomputed light tiles to improve coherence.

        // Initial candidate sampling options.
        uint32_t localLightCandidateCount = 24;   ///< Number of initial local light candidate samples.
        uint32_t infiniteLightCandidateCount = 8; ///< Number of initial infinite light candidate samples.
        uint32_t envLightCandidateCount = 8;      ///< Number of initial environment light candidate samples.
        uint32_t brdfCandidateCount = 1;          ///< Number of initial brdf candidate samples.
        float brdfCutoff = 0.f;              ///< Value in range[0, 1] to determine how much to shorten BRDF rays. 0 to disable shortening.
        bool testCandidateVisibility = true; ///< Test visibility on selected candidate sample before doing resampling.

        // Resampling options.
        BiasCorrection biasCorrection = BiasCorrection::RayTraced; ///< Bias correction mode.
        float depthThreshold = 0.1f;  ///< Relative depth difference at which pixels are classified too far apart to be reused (0.1 = 10%).
        float normalThreshold = 0.5f; ///< Cosine of the angle between normals, below which pixels are classified too far apart to be
                                      ///< reused.

        // Spatial resampling options.
        float samplingRadius = 30.0f;    ///< Screen-space radius for spatial resampling, measured in pixels.
        uint32_t spatialSampleCount = 1; ///< Number of neighbor pixels considered for resampling.
        uint32_t spatialIterations = 5;  ///< Number of spatial resampling passes (only used in SpatialResampling mode, Spatiotemporal mode
                                         ///< always uses 1 iteration).

        // Temporal resampling options.
        uint32_t maxHistoryLength = 20;     ///< Maximum history length for temporal reuse, measured in frames.
        float boilingFilterStrength = 0.0f; ///< 0 = off, 1 = full strength.

        // Rendering options.
        float rayEpsilon = 1.0e-3f; ///< Ray epsilon for avoiding self-intersection of visibility rays.

        // Parameter controlling behavior of final shading. Lights can have an emissive texture containing arbitrarily high frequncies.
        // To improve convergence and significantly reduce texture lookup costs, this code always uses a preintegrated emissivity over each
        // triangle during resampling. This preintegrated value can *also* be used for final shading, which reduces noise (quite a lot if
        // the artists go crazy on the textures), at the tradeoff of losing high frequency details in the lighting. We recommend using
        // "false" -- if that overblurs, consider tessellating lights to better match lighting variations. If dynamic textured lights make
        // textured lookups vital for quality, we recommend enabling this setting on only lights where it is vital.
        bool useEmissiveTextures = false; ///< Use emissive textures to return final sample incident radiance (true is slower and noisier).

        // ReSTIRGDI options not currently exposed in the sample, as it isn't configured for them to produce acceptable results.

        bool enableVisibilityShortcut = false;  ///< Reuse visibility across frames to reduce cost; requires careful setup to avoid bias /
                                                ///< numerical blowups.
        bool enablePermutationSampling = false; ///< Enables permuting the pixels sampled from the previous frame (noisier but more denoiser
                                                ///< friendly).

        // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
        Options() {}

        template<typename Archive>
        void serialize(Archive& ar)
        {
            ar("mode", mode);

            ar("presampledTileCount", presampledTileCount);
            ar("presampledTileSize", presampledTileSize);
            ar("storeCompactLightInfo", storeCompactLightInfo);

            ar("localLightCandidateCount", localLightCandidateCount);
            ar("infiniteLightCandidateCount", infiniteLightCandidateCount);
            ar("envLightCandidateCount", envLightCandidateCount);
            ar("brdfCandidateCount", brdfCandidateCount);
            ar("brdfCutoff", brdfCutoff);
            ar("testCandidateVisibility", testCandidateVisibility);

            ar("biasCorrection", biasCorrection);
            ar("depthThreshold", depthThreshold);
            ar("normalThreshold", normalThreshold);

            ar("samplingRadius", samplingRadius);
            ar("spatialSampleCount", spatialSampleCount);
            ar("spatialIterations", spatialIterations);

            ar("maxHistoryLength", maxHistoryLength);
            ar("boilingFilterStrength", boilingFilterStrength);

            ar("rayEpsilon", rayEpsilon);

            ar("useEmissiveTextures", useEmissiveTextures);

            ar("enableVisibilityShortcut", enableVisibilityShortcut);
            ar("enablePermutationSampling", enablePermutationSampling);
        }
    };

    static_assert(std::is_trivially_copyable<Options>(), "Options needs to be trivially copyable");

    /** Constructor.
        \param[in] pScene Scene.
        \param[in] options Configuration options.
    */
    ReSTIRGDI(ref<IScene> pScene, const Options& options = Options());

    /** Set the configuration options.
        \param[in] options Configuration options.
    */
    void setOptions(const Options& options);

    /** Returns the current configuration options.
        \return The configuration options.
    */
    const Options& getOptions() { return mOptions; }

    /** Render the GUI.
        \return True if options were changed, false otherwise.
    */
    bool renderUI(Gui::Widgets& widget);

    /** Get a list of shader defines for using the ReSTIRGDI sampler.
        \return List of shader defines.
    */
    DefineList getDefines() const;

    /** Bind the ReSTIRGDI sampler to a given shader var.
        Note: ReSTIRGDI is always bound to the global "gRESTIRGDI" variable, so we expect a root shader variable here.
        \param[in] rootVar The root shader variable to set the data into.
    */
    void bindShaderData(const ShaderVar& rootVar);

    /** Begin a frame.
        Must be called once at the beginning of each frame.
        \param[in] pRenderContext Render context.
        \param[in] frameDim Current frame dimension.
    */
    void beginFrame(RenderContext* pRenderContext, const uint2& frameDim);

    /** End a frame.
        Must be called one at the end of each frame.
        \param[in] pRenderContext Render context.
    */
    void endFrame(RenderContext* pRenderContext);

    /** Update and run this frame's ReSTIRGDI resampling, allowing final samples to be queried afterwards.
        Must be called once between beginFrame() and endFrame().
        \param[in] pRenderContext Render context.
        \param[in] pMotionVectors Motion vectors for temporal reprojection.
    */
    void update(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors);

    /** Get the pixel debug component.
        \return Returns the pixel debug component.
    */
    PixelDebug& getPixelDebug() { return *mpPixelDebug; }

private:
    ref<IScene> mpScene;  ///< Scene (set on initialization).
    ref<Device> mpDevice; ///< GPU device.
    Options mOptions;     ///< Configuration options.

    std::unique_ptr<PixelDebug> mpPixelDebug; ///< Pixel debug component.

    sigs::Connection mUpdateFlagsConnection; ///< Connection to the UpdateFlags signal.
    /// IScene::UpdateFlags accumulated since last `beginFrame()`
    IScene::UpdateFlags mUpdateFlags = IScene::UpdateFlags::None;

    // If the SDK is not installed, we leave out most of the implementation.

    // ReSTIRGDI state.

    restir::ContextParameters mRESTIRContextParams;         ///< Parameters that largely stay constant during program execution.
    RESTIR_ResamplingRuntimeParameters mRESTIRShaderParams; ///< Structure passed to the GPU per frame.
    std::unique_ptr<restir::Context> mpRESTIRContext;       ///< The ReSTIRGDI context.

    // Runtime state.

    uint mFrameIndex = 0;                    ///< Current frame index.
    uint2 mFrameDim = {0, 0};                ///< Current frame dimension in pixels.
    uint32_t mLastFrameReservoirID = 1;      ///< Index of the reservoir containing last frame's output (for temporal reuse).
    uint32_t mCurrentSurfaceBufferIndex = 0; ///< Index of the surface buffer used for the current frame (0 or 1).

    CameraData mPrevCameraData; ///< Previous frame's camera data.

    // ReSTIRGDI categorizes lights into local, infinite and an environment light.
    // Falcor has emissive (triangle) lights, analytic lights (point, area, directional, distant) and an environment light.
    // This struct keeps track of the mapping from Falcor lights to ReSTIRGDI lights.
    // Falcors emissive lights and (local) analytic lights generate ReSTIRGDIs local lights.
    // Falcors directional and distant lights generate ReSTIRGDIs infinite lights.
    struct
    {
        uint32_t emissiveLightCount = 0;         ///< Total number of local emissive lights (triangle lights).
        uint32_t localAnalyticLightCount = 0;    ///< Total number of local analytic lights (point lights).
        uint32_t infiniteAnalyticLightCount = 0; ///< Total number of infinite analytic lights (directional and distant lights).
        bool envLightPresent = false;            ///< True if environment light is present.

        uint32_t prevEmissiveLightCount = 0;      ///< Total number of local emissive lights (triangle lights) in previous frame.
        uint32_t prevLocalAnalyticLightCount = 0; ///< Total number of local analytic lights (point lights) in previous frame.

        std::vector<uint32_t> analyticLightIDs; ///< List of analytic light IDs sorted for use with ReSTIRGDI.

        uint32_t getLocalLightCount() const { return emissiveLightCount + localAnalyticLightCount; }
        uint32_t getInfiniteLightCount() const { return infiniteAnalyticLightCount; }
        uint32_t getTotalLightCount() const { return getLocalLightCount() + getInfiniteLightCount() + (envLightPresent ? 1 : 0); }

        uint32_t getFirstLocalLightIndex() const { return 0; }
        uint32_t getFirstInfiniteLightIndex() const { return getLocalLightCount(); }
        uint32_t getEnvLightIndex() const { return getLocalLightCount() + getInfiniteLightCount(); }
    } mLights;

    // Flags for triggering various actions and updates.

    struct
    {
        bool updateEmissiveLights = true;     ///< Set if emissive triangles have changed (moved, enabled/disabled).
        bool updateEmissiveLightsFlux = true; ///< Set if emissive triangles have changed intensities.
        bool updateAnalyticLights = true;     ///< Set if analytic lights have changed (enabled/disabled).
        bool updateAnalyticLightsFlux = true; ///< Set if analytic lights have changed intensities.
        bool updateEnvLight = true;           ///< Set if environment light has changed (env map, intensity, enabled/disabled).
        bool recompileShaders = true;         ///< Set if shaders need recompilation on next beginFrame() call.
        bool clearReservoirs = false;         ///< Set if reservoirs need to be cleared on next beginFrame() call (useful when changing
                                              ///< configuration).
    } mFlags;

    // Resources.

    ref<Buffer> mpAnalyticLightIDBuffer;     ///< Buffer storing a list of analytic light IDs used in the scene.
    ref<Buffer> mpLightInfoBuffer;           ///< Buffer storing information about all the lights in the scene.
    ref<Texture> mpLocalLightPdfTexture;     ///< Texture storing the PDF for sampling local lights proportional to radiant flux.
    ref<Texture> mpEnvLightLuminanceTexture; ///< Texture storing luminance of the environment light.
    ref<Texture> mpEnvLightPdfTexture; ///< Texture storing the PDF for sampling the environment light proportional to luminance (times
                                       ///< solid angle).

    ref<Buffer> mpLightTileBuffer; ///< Buffer storing precomputed light tiles (see presampleLights()). This is called "ris buffer" in
                                   ///< ReSTIRGDI.
    ref<Buffer> mpCompactLightInfoBuffer; ///< Optional buffer storing compact light info for samples in the light tile buffer for improved
                                          ///< coherence.

    ref<Buffer> mpReservoirBuffer;       ///< Buffer storing light reservoirs between kernels (and between frames)
    ref<Buffer> mpSurfaceDataBuffer;     ///< Buffer storing the surface data for the current and previous frames.
    ref<Buffer> mpNeighborOffsetsBuffer; ///< Buffer storing a poisson(-ish) distribution of offsets for sampling randomized neighbors.

    // Compute passes.

    // Passes to pipe data from Falcor into ReSTIRGDI.

    ref<ComputePass> mpReflectTypes;       ///< Helper pass for reflecting type information.
    ref<ComputePass> mpUpdateLightsPass;   ///< Update the light infos and light PDF texture.
    ref<ComputePass> mpUpdateEnvLightPass; ///< Update the environment light luminance and PDF texture.

    // Passes for all ReSTIRGDI modes.

    ref<ComputePass> mpPresampleLocalLightsPass;    ///< Presample local lights into light tiles.
    ref<ComputePass> mpPresampleEnvLightPass;       ///< Presample the environment light into light tiles.
    ref<ComputePass> mpGenerateCandidatesPass;      ///< Generate initial candidates.
    ref<ComputePass> mpTestCandidateVisibilityPass; ///< Test visibility for selected candidate.

    // Passes for various types of reuse.

    ref<ComputePass> mpSpatialResamplingPass;        ///< Spatial only resampling.
    ref<ComputePass> mpTemporalResamplingPass;       ///< Temporal only resampling.
    ref<ComputePass> mpSpatiotemporalResamplingPass; ///< Spatiotemporal resampling.

    // Compute pass launches.

    void bindShaderDataInternal(const ShaderVar& rootVar, const ref<Texture>& pMotionVectors, bool bindScene = true);
    void updateLights(RenderContext* pRenderContext);
    void updateEnvLight(RenderContext* pRenderContext);
    void presampleLights(RenderContext* pRenderContext);
    void generateCandidates(RenderContext* pRenderContext, uint32_t outputReservoirID);
    void testCandidateVisibility(RenderContext* pRenderContext, uint32_t candidateReservoirID);
    uint32_t spatialResampling(RenderContext* pRenderContext, uint32_t inputReservoirID);
    uint32_t temporalResampling(
        RenderContext* pRenderContext,
        const ref<Texture>& pMotionVectors,
        uint32_t candidateReservoirID,
        uint32_t lastFrameReservoirID
    );
    uint32_t spatiotemporalResampling(
        RenderContext* pRenderContext,
        const ref<Texture>& pMotionVectors,
        uint32_t candidateReservoirID,
        uint32_t lastFrameReservoirID
    );

    // Internal routines.

    void loadShaders();
    void prepareResources(RenderContext* pRenderContext);
    void setReSTIRGDIFrameParameters();
};

FALCOR_ENUM_REGISTER(ReSTIRGDI::Mode);
FALCOR_ENUM_REGISTER(ReSTIRGDI::BiasCorrection);
}
