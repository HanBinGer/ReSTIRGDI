add_plugin(ReSTIRGDIPass)

target_sources(ReSTIRGDIPass PRIVATE
    ReSTIRGDIPass.cpp
    ReSTIRGDIPass.h
    ReSTIRGDI.h
    FinalShading.cs.slang
    LoadShadingData.slang
    PrepareSurfaceData.cs.slang
)

target_copy_shaders(ReSTIRGDIPass RenderPasses/ReSTIRGDIPass)

target_source_group(ReSTIRGDIPass "RenderPasses")
