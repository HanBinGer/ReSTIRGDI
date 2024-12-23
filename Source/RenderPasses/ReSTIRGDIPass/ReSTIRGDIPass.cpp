/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "ReSTIRGDIPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

using namespace Falcor;

namespace
{
const std::string kPrepareSurfaceDataFile = "RenderPasses/ReSTIRGDIPass/PrepareSurfaceData.cs.slang";
const std::string kFinalShadingFile = "RenderPasses/ReSTIRGDIPass/FinalShading.cs.slang";

const std::string kInputVBuffer = "vbuffer";
const std::string kInputTexGrads = "texGrads";
const std::string kInputMotionVectors = "mvec";

const Falcor::ChannelList kInputChannels = {
    // clang-format off
    { kInputVBuffer,            "gVBuffer",                 "Visibility buffer in packed format"                       },
    { kInputTexGrads,           "gTextureGrads",            "Texture gradients", true /* optional */                   },
    { kInputMotionVectors,      "gMotionVector",            "Motion vector buffer (float format)", true /* optional */ },
    // clang-format on
};

const Falcor::ChannelList kOutputChannels = {
    // clang-format off
    { "color",                  "gColor",                   "Final color",              true /* optional */, ResourceFormat::RGBA32Float },
    { "emission",               "gEmission",                "Emissive color",           true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseIllumination",    "gDiffuseIllumination",     "Diffuse illumination",     true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseReflectance",     "gDiffuseReflectance",      "Diffuse reflectance",      true /* optional */, ResourceFormat::RGBA32Float },
    { "specularIllumination",   "gSpecularIllumination",    "Specular illumination",    true /* optional */, ResourceFormat::RGBA32Float },
    { "specularReflectance",    "gSpecularReflectance",     "Specular reflectance",     true /* optional */, ResourceFormat::RGBA32Float },
    // clang-format on
};

// Scripting options.
const char* kOptions = "options";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRGDIPass>();
}

ReSTIRGDIPass::ReSTIRGDIPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);
}

Properties ReSTIRGDIPass::getProperties() const
{
    Properties props;
    props[kOptions] = mOptions;
    return props;
}

void ReSTIRGDIPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kOptions)
            mOptions = value;
        else
            logWarning("Unknown property '{}' in ReSTIRGDIPass properties.", key);
    }
}

RenderPassReflection ReSTIRGDIPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassOutputs(reflector, kOutputChannels);
    addRenderPassInputs(reflector, kInputChannels);
    return reflector;
}

void ReSTIRGDIPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Clear outputs if no scene is loaded.
    if (!mpScene)
    {
        clearRenderPassChannels(pRenderContext, kOutputChannels, renderData);
        return;
    }

    // Check for scene changes that require shader recompilation.
    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        recreatePrograms();
    }

    FALCOR_ASSERT(mpReSTIRGDI);

    const auto& pVBuffer = renderData.getTexture(kInputVBuffer);
    const auto& pMotionVectors = renderData.getTexture(kInputMotionVectors);

    auto& dict = renderData.getDictionary();

    // Update refresh flag if changes that affect the output have occured.
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    mGBufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);

    mpReSTIRGDI->beginFrame(pRenderContext, mFrameDim);

    prepareSurfaceData(pRenderContext, pVBuffer);

    mpReSTIRGDI->update(pRenderContext, pMotionVectors);

    finalShading(pRenderContext, pVBuffer, renderData);

    mpReSTIRGDI->endFrame(pRenderContext);
}

void ReSTIRGDIPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpReSTIRGDI = nullptr;

    recreatePrograms();

    if (mpScene)
    {
        if (pScene->hasProceduralGeometry())
        {
            logWarning("ReSTIRGDIPass: This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        mpReSTIRGDI = std::make_unique<ReSTIRGDI>(mpScene, mOptions);
    }
}

bool ReSTIRGDIPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpReSTIRGDI ? mpReSTIRGDI->getPixelDebug().onMouseEvent(mouseEvent) : false;
}

void ReSTIRGDIPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void ReSTIRGDIPass::renderUI(Gui::Widgets& widget) {
    if (mpReSTIRGDI)
    {
        mOptionsChanged = mpReSTIRGDI->renderUI(widget);
        if (mOptionsChanged)
            mOptions = mpReSTIRGDI->getOptions();
    }
}

void ReSTIRGDIPass::recreatePrograms()
{
    mpPrepareSurfaceDataPass = nullptr;
    mpFinalShadingPass = nullptr;
}

void ReSTIRGDIPass::prepareSurfaceData(RenderContext* pRenderContext, const ref<Texture>& pVBuffer)
{
    FALCOR_ASSERT(mpReSTIRGDI);
    FALCOR_ASSERT(pVBuffer);

    FALCOR_PROFILE(pRenderContext, "prepareSurfaceData");

    if (!mpPrepareSurfaceDataPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPrepareSurfaceDataFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        defines.add(mpReSTIRGDI->getDefines());
        defines.add("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

        mpPrepareSurfaceDataPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mpPrepareSurfaceDataPass->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

    auto rootVar = mpPrepareSurfaceDataPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    mpReSTIRGDI->bindShaderData(rootVar);

    auto var = rootVar["gPrepareSurfaceData"];
    var["vbuffer"] = pVBuffer;
    var["frameDim"] = mFrameDim;

    mpPrepareSurfaceDataPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void ReSTIRGDIPass::finalShading(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const RenderData& renderData)
{
    FALCOR_ASSERT(mpReSTIRGDI);
    FALCOR_ASSERT(pVBuffer);

    FALCOR_PROFILE(pRenderContext, "finalShading");

    if (!mpFinalShadingPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kFinalShadingFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        defines.add(mpReSTIRGDI->getDefines());
        defines.add("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");
        defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
        defines.add(getValidResourceDefines(kOutputChannels, renderData));

        mpFinalShadingPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mpFinalShadingPass->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");
    mpFinalShadingPass->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mpFinalShadingPass->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    auto rootVar = mpFinalShadingPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    mpReSTIRGDI->bindShaderData(rootVar);

    auto var = rootVar["gFinalShading"];
    var["vbuffer"] = pVBuffer;
    var["frameDim"] = mFrameDim;

    // Bind output channels as UAV buffers.
    auto bind = [&](const ChannelDesc& channel)
    {
        ref<Texture> pTex = renderData.getTexture(channel.name);
        rootVar[channel.texname] = pTex;
    };
    for (const auto& channel : kOutputChannels)
        bind(channel);

    mpFinalShadingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}
