from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")

    gVBufferParams = {
        'samplePattern': "Center",
        'sampleCount': 1,
        'useAlphaTest': True,
        'subPixelRandom' : "UnitQuad",
        'useDOF' : True
    }
    gPathTracerParams = {
        'samplesPerPixel': 1,
        'useReSTIR': True,
        'emissiveSampler': "Power",
        'maxSurfaceBounces' : 0,
        'maxDiffuseBounces' : 0,
        'maxSpecularBounces' : 0,
        'maxTransmissionBounces' : 0,
        'disableCaustics' : True,
        'ReSTIRGDIOptions' : {
            "resampleEmissionMode" : "None", "useTemporalResampling" : True, "useSpatialResampling" : True,
            "temporalMode" : "FractionalMotion_Shifting2RIS",
            "temporalShiftMappingModeRIS1" : "OnlyRandomReplay", "temporalShiftMappingModeRIS2" : "MIS", "spatialShiftMappingMode" : "MIS",
            "optimizeShift2RIS" : False,
        }
    }

    VBufferRT = createPass("VBufferRT", gVBufferParams)
    g.addPass(VBufferRT, "VBufferRT")
    PathTracer = createPass("PathTracer", gPathTracerParams)
    g.addPass(PathTracer, "PathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.depth", "PathTracer.depth")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")

    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

pauseClock = True
if pauseClock:
    m.clock.stop()
