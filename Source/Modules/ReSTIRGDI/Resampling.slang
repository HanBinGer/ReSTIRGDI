/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Reservoir;
__exported import EvalContext;
__exported import Utils.Sampling.SampleGeneratorInterface;
import Scene.Scene;
import Scene.Camera.CameraData;
import Scene.RaytracingInline;
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;
import Utils.Geometry.GeometryHelpers;
import Lights;
import LoadShadingData;
import Params;

// Some utility funtions shared between shaders
uint getPixelIndex(uint2 frameDim, uint2 pixel, uint passIdx)
{
    uint PixelIdxOnScreen = pixel.y * frameDim.x + pixel.x;
    uint totalScreenPixels = frameDim.x * frameDim.y;

    return totalScreenPixels * passIdx + // which pass?
           PixelIdxOnScreen;             // which pixel on screen?
}

// static const bool kUseAlphaTest = USE_ALPHA_TEST;
static const float kFloatEpsilon = 0.0001f;

cbuffer SharedResamplingCB
{
    bool useRandomThreshold;
    bool reuseThresholdOffset;
    bool scaleTwoShiftsWeightForMIS;
    bool betterScaleFuntionForMIS;
    uint scalingFunctionIndex;
}

/** Resampled importance sampling state.
*/
struct RisState
{
    LightSample lightSample;    ///< Selected light sample.
    uint pathSample;
    float weightSum;            ///< Sum of weights seen so far.
    float M;                    ///< Number of samples seen so far (float required for pairwise MIS).
    float weight;               ///< UCW weight
    float targetPdf;
    float canonicalWeight; ///< Canonical sample weight factor (required for pairwise MIS).

    /** Create an empty RIS state.
     */
    static RisState createEmpty(uint pathSample = 2u)
    {
        RisState state = {};
        state.lightSample = LightSample::createInvalid();
        state.pathSample = pathSample;
        return state;
    }

    static RisState createEmpty(const Reservoir reservoir)
    {
        RisState state = {};
        state.lightSample = LightSample::createInvalid();
        state.pathSample = reservoir.pathSample;
        return state;
    }

    /** Create a reservoir from the current RIS state.
        An empty reservoir is returned if the weight is infinite or NaN.
        \return Returns a reservoir representing the current RIS state.
    */
    Reservoir toReservoir()
    {
        Reservoir reservoir;
        reservoir.lightSample = lightSample;
        reservoir.weight = weight;
        reservoir.M = uint(M);
        reservoir.pathSample = pathSample;
        reservoir.targetPdf = targetPdf;
        if (isinf(reservoir.weight) || isnan(reservoir.weight)) reservoir = Reservoir::createEmpty(pathSample);
        return reservoir;
    }
};


/** Resample a light sample.
    This implements Algorithm 2 from the paper.
    \param[in,out] state RIS state.
    \param[in] lightSample Light sample to resample.
    \param[in] targetPdf Pdf of the sample in the target distribution (p-hat).
    \param[in] sourcePdf Pdf of the sample in the source distribution (p).
    \param[in,out] sg Sample generator.
    \return Returns true if new sample was selected.
*/
bool streamingInitialResampleMIS<S : ISampleGenerator>(inout RisState state, const LightSample lightSample, float targetPdf, float sourcePdf, float misWeight, float pHat_withoutAlbedo,
    inout S sg)
{
    float sampleWeight = misWeight * targetPdf / sourcePdf;

    state.weightSum += sampleWeight;
    state.M += 1.f;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.lightSample = lightSample;
        state.targetPdf = targetPdf;
    }

    return selectSample;
}

// No pHat_withoutAlbedo
bool streamingInitialResampleMIS<S : ISampleGenerator>( inout RisState state, const LightSample lightSample, float targetPdf, float sourcePdf, float misWeight, inout S sg )
{
    float sampleWeight = misWeight * targetPdf / sourcePdf;

    state.weightSum += sampleWeight;
    state.M += 1.f;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.lightSample = lightSample;
        state.targetPdf = targetPdf;
    }

    return selectSample;
}

// Resample a path sample
bool streamingInitialResampleMIS<S : ISampleGenerator>(inout RisState state, uint pathSample, float targetPdf, float sourcePdf, float misWeight, inout S sg)
{
    float sampleWeight = misWeight * targetPdf / sourcePdf;
    state.weightSum += sampleWeight;
    state.M += 1.f;
    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.pathSample = pathSample;
        state.targetPdf = targetPdf;
    }

    return selectSample;
}

// For RGB reservoirs: resample with a reservoir using balanced heuristic
bool streamingResampleStepMIS<S : ISampleGenerator>(inout RisState state, const Reservoir reservoir, inout S sg, float misWeight)
{
    float sampleWeight = misWeight * reservoir.targetPdf * reservoir.weight * 1.0f;

    state.weightSum += sampleWeight;
    state.M += reservoir.M;

    print("sampleWeight = ", sampleWeight);
    print("reservoir UCW = ", reservoir.weight);

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.lightSample = reservoir.lightSample;
        state.targetPdf = reservoir.targetPdf;
        state.pathSample = reservoir.pathSample;
    }

    return selectSample;
}

// --------------------------------------------
// New resampling parts
// --------------------------------------------

float mFactor(float q0, float q1)
{
    return q0 == 0.f ? 1.f : clamp(pow(min(q1 / q0, 1.f), 8.f), 0.f, 1.f);
}

float pairwiseMIS_nonDefensiveNonCanonical(float mSum, float mCandidate, float pi, float mCanonical, float pc, float out_J = 1.0f)
{
    float pHatFrom_i = pi * out_J;
    return pc == 0.0f ? 0.0f : mCandidate * pHatFrom_i / ((mSum - mCanonical) * pHatFrom_i + mCanonical * pc);
}

float pairwiseMIS_nonDefensiveCanonical(float mSum, float mCandidate, float pi, float mCanonical, float pc, float out_J = 1.0f)
{
    float w = mCanonical * pc;
    float pHatFrom_i = pi * out_J;
    return pc == 0.0f ? 0.0f : (mCandidate / (mSum - mCanonical)) * w / ((mSum - mCanonical) * pHatFrom_i + w);
}

float pairwiseMIS_defensiveNonCanonical(float mSum, float mCandidate, float pi, float mCanonical, float pc, float out_J = 1.0f)
{
    float pHatFrom_i = pi * out_J;
    float w = (mSum - mCanonical) * pHatFrom_i;
    return pc == 0.0f ? 0.0f : (mCandidate / mSum) * w / (w + mCanonical * pc);
}

float pairwiseMIS_defensiveCanonical(float mSum, float mCandidate, float pi, float mCanonical, float pc, float out_J = 1.0f)
{
    float pHatFrom_i = pi * out_J;
    float w = mCanonical * pc;
    return pc == 0.0f ? 0.0f : (mCandidate / mSum) * w / ((mSum - mCanonical) * pHatFrom_i + w);
}

// For origin point reservoir temporal resampling
bool temporalResamplePointReservoirPairwiseMIS<S : ISampleGenerator>(
    inout RisState state,
    const Reservoir canonicalReservoir, const EvalContext canonicalContext, const Reservoir candidateReservoir, const EvalContext candidateContext,
    const Lights lights, const bool useMFactor, const float confidenceWeightSum, const float intersectAreaSize,
     inout S sg, const bool withVisibility = false)
{
    float canonicalTargetPdf = canonicalReservoir.targetPdf;
    float candidateTargetPdf = candidateReservoir.targetPdf;

    // Candidate sample evaluated at current
    const LoadedLightSample candidateLightSampleAtCurr = lights.loadLightSample(candidateReservoir.lightSample);
    float candidateTargetPdfAtOther = canonicalContext.evalTargetFunction(candidateLightSampleAtCurr, lights, withVisibility, sg, 2u);

    // Canonical sample evaluated at previous
    const LoadedLightSample canonicalLightSampleAtPrev = lights.loadLightSample(canonicalReservoir.lightSample);
    float canonicalTargetPdfAtOther = candidateContext.evalTargetFunction( canonicalLightSampleAtPrev, lights, withVisibility, sg, 2u);

    // Compute MIS weights
    float candidateConfidenceWeight = candidateReservoir.M * intersectAreaSize;
    float m0 = pairwiseMIS_nonDefensiveNonCanonical(confidenceWeightSum, candidateConfidenceWeight, candidateTargetPdf,
        canonicalReservoir.M, candidateTargetPdfAtOther);
    float m1 = pairwiseMIS_nonDefensiveCanonical(confidenceWeightSum, candidateConfidenceWeight, canonicalTargetPdfAtOther,
        canonicalReservoir.M, canonicalTargetPdf);

    // Candidate resampling weight
    float sampleWeight = candidateTargetPdfAtOther * candidateReservoir.weight * m0;
    float mScaler = min(mFactor(candidateTargetPdf, candidateTargetPdfAtOther), mFactor(canonicalTargetPdfAtOther, canonicalTargetPdf));
    state.M += candidateConfidenceWeight * (useMFactor ? mScaler : 1.0f);
    state.weightSum += sampleWeight;
    state.canonicalWeight += m1; // MIS weight for canonical reservoir

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.lightSample = candidateReservoir.lightSample;
        state.targetPdf = candidateTargetPdfAtOther;
    }

    return selectSample;
}

// For origin point reservoir spatial resampling
bool spatialResamplePointReservoirPairwiseMIS<S : ISampleGenerator>(
    inout RisState state,
    const Reservoir canonicalReservoir, const EvalContext canonicalContext, const Reservoir candidateReservoir, const EvalContext candidateContext,
    const Lights lights, const bool useMFactor, const float confidenceWeightSum, inout S sg,
    const bool withVisibility = false)
{
    float canonicalTargetPdf = canonicalReservoir.targetPdf;
    float candidateTargetPdf = candidateReservoir.targetPdf;

    const LoadedLightSample candidateLightSample = lights.loadLightSample(candidateReservoir.lightSample);
    const LoadedLightSample canonicalLightSample = lights.loadLightSample(canonicalReservoir.lightSample);

    float candidateTargetPdfAtOther = canonicalContext.evalTargetFunction( candidateLightSample, lights, withVisibility, sg, 2u); // Pc(Xi)
    float canonicalTargetPdfAtOther = candidateContext.evalTargetFunction( canonicalLightSample, lights, withVisibility, sg, 2u); // Pi(Xc)

    // Compute MIS weights
    float candidateConfidenceWeight = candidateReservoir.M;
    float m0 = pairwiseMIS_defensiveNonCanonical(confidenceWeightSum, candidateConfidenceWeight, candidateTargetPdf, canonicalReservoir.M, candidateTargetPdfAtOther);
    float m1 = pairwiseMIS_defensiveCanonical(confidenceWeightSum, candidateConfidenceWeight, canonicalTargetPdfAtOther, canonicalReservoir.M, canonicalTargetPdf);

    // Candidate resampling weight
    float sampleWeight = candidateTargetPdfAtOther * candidateReservoir.weight * m0;
    float mScaler = min(mFactor(candidateTargetPdf, candidateTargetPdfAtOther), mFactor(canonicalTargetPdfAtOther, canonicalTargetPdf));
    state.M += candidateConfidenceWeight * (useMFactor ? mScaler : 1.0f);
    state.weightSum += sampleWeight;
    state.canonicalWeight += m1; // MIS weight for canonical reservoir

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.lightSample = candidateReservoir.lightSample;
        state.targetPdf = candidateTargetPdfAtOther;
    }

    return selectSample;
}

// Resample the canonical reservoir in pairwise MIS -> jacobian is just one (identical shift)
bool streamingResampleFinalizeMis<S : ISampleGenerator>( inout RisState state, const Reservoir canonicalReservoir, float canonicalTargetPdf, inout S sg)
{
    float sampleWeight = state.canonicalWeight * canonicalTargetPdf * canonicalReservoir.weight * 1.0f;

    state.M += canonicalReservoir.M;
    state.weightSum += sampleWeight;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;
    if (selectSample)
    {
        state.targetPdf = canonicalTargetPdf;
        state.lightSample = canonicalReservoir.lightSample;
        state.pathSample = canonicalReservoir.pathSample;
    }

    return selectSample;
}

