#pragma once
#include "Core/Macros.h"
#include "Core/Enum.h"
#include "Utils/Properties.h"
#include "Utils/Debug/PixelDebug.h"
#include "Scene/Scene.h"
#include <memory>
#include <type_traits>
#include <vector>

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
        Mode mode = Mode::SpatiotemporalResampling; ///< RTXDI sampling mode.

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

        // RTXDI options not currently exposed in the sample, as it isn't configured for them to produce acceptable results.

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

    /** Get a list of shader defines for using the RTXDI sampler.
        \return List of shader defines.
    */
    DefineList getDefines() const;

    /** Bind the RTXDI sampler to a given shader var.
        Note: RTXDI is always bound to the global "gRTXDI" variable, so we expect a root shader variable here.
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

    /** Update and run this frame's RTXDI resampling, allowing final samples to be queried afterwards.
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

};

FALCOR_ENUM_REGISTER(ReSTIRGDI::Mode);
FALCOR_ENUM_REGISTER(ReSTIRGDI::BiasCorrection);
}
