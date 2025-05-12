add_plugin(PathTracer)

target_sources(PathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    NRDHelpers.slang
    Params.slang
    PathState.slang
    PathTracer.slang
    PathTracer.cpp
    PathTracer.h
    PathTracerNRD.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_link_libraries(PathTracer PUBLIC ReSTIRGDI)
# add_dependencies(PathTracer ReSTIRGDI)

target_source_group(PathTracer "RenderPasses")

# ReSTIRGDI source files
# Note: These added after calling target_source_group macro as they are outside the current folder.
target_sources(PathTracer PRIVATE
    ../../Modules/ReSTIRGDI/EvalContext.slang
    ../../Modules/ReSTIRGDI/EvaluateFinalSamples.cs.slang
    ../../Modules/ReSTIRGDI/FinalSample.slang
    ../../Modules/ReSTIRGDI/GenerateLightTiles.cs.slang
    ../../Modules/ReSTIRGDI/InitialResampling.cs.slang
    ../../Modules/ReSTIRGDI/Lights.slang
    ../../Modules/ReSTIRGDI/LightSample.slang
    ../../Modules/ReSTIRGDI/LocalEmissiveTriangle.slang
    ../../Modules/ReSTIRGDI/Params.slang
    ../../Modules/ReSTIRGDI/ReflectTypes.cs.slang
    ../../Modules/ReSTIRGDI/Resampling.slang
    ../../Modules/ReSTIRGDI/Reservoir.slang
    ../../Modules/ReSTIRGDI/ReSTIRGDI.slang
    ../../Modules/ReSTIRGDI/SpatialResampling.cs.slang
    ../../Modules/ReSTIRGDI/SurfaceData.slang
    ../../Modules/ReSTIRGDI/TemporalResampling.cs.slang
    ../../Modules/ReSTIRGDI/UpdateEmissiveTriangles.cs.slang
    ../../Modules/ReSTIRGDI/LoadShadingData.slang
    ../../Modules/ReSTIRGDI/SimpleCameraData.slang
    ../../Modules/ReSTIRGDI/StaticParams.slang
)

target_copy_shaders(PathTracer RenderPasses/PathTracer)
