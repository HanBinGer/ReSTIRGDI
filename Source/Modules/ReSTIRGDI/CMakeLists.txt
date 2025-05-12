add_library(ReSTIRGDI STATIC)
# add_plugin(ReSTIRGDI)

target_sources(ReSTIRGDI PRIVATE
    EvalContext.slang
    EvaluateFinalSamples.cs.slang
    FinalSample.slang
    GenerateLightTiles.cs.slang
    InitialResampling.cs.slang
    Lights.slang
    LightSample.slang
    LocalEmissiveTriangle.slang
    Params.slang
    ReflectTypes.cs.slang
    Resampling.slang
    Reservoir.slang
    ReSTIRGDI.cpp
    ReSTIRGDI.h
    ReSTIRGDI.slang
    SpatialResampling.cs.slang
    SurfaceData.slang
    TemporalResampling.cs.slang
    UpdateEmissiveTriangles.cs.slang
    LoadShadingData.slang
    SimpleCameraData.slang
    StaticParams.slang
)

target_copy_shaders(ReSTIRGDI Modules/ReSTIRGDI)

target_link_libraries(ReSTIRGDI
    PUBLIC
        Falcor
)

target_include_directories(ReSTIRGDI
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(ReSTIRGDI
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
)

target_source_group(ReSTIRGDI "Modules")

validate_headers(ReSTIRGDI)
