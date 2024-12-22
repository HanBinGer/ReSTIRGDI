from falcor import *

def render_graph_RESTIRGDI():
    g = RenderGraph("RESTIRGDI")
    VBufferRT = createPass("VBufferRT")
    g.addPass(VBufferRT, "VBufferRT")
    ReSTIRGDIPass = createPass("ReSTIRGDIPass")
    g.addPass(ReSTIRGDIPass, "ReSTIRGDIPass")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "ReSTIRGDIPass.vbuffer")
    g.addEdge("VBufferRT.mvec", "ReSTIRGDIPass.mvec")
    g.addEdge("ReSTIRGDIPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

RESTIRGDI = render_graph_RESTIRGDI()
try: m.addGraph(RESTIRGDI)
except NameError: None
