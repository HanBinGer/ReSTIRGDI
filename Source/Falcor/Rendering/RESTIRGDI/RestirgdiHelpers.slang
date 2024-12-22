#ifndef RESTIR_HELPERS_SLANG
#define RESTIR_HELPERS_SLANG

import RestirgdiMath;

uint RESTIR_ReservoirPositionToPointer(
    RESTIR_ResamplingRuntimeParameters params,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint2 blockIdx = reservoirPosition / RESTIR_RESERVOIR_BLOCK_SIZE;
    uint2 positionInBlock = reservoirPosition % RESTIR_RESERVOIR_BLOCK_SIZE;

    return reservoirArrayIndex * params.reservoirArrayPitch
        + blockIdx.y * params.reservoirBlockRowPitch
        + blockIdx.x * (RESTIR_RESERVOIR_BLOCK_SIZE * RESTIR_RESERVOIR_BLOCK_SIZE)
        + positionInBlock.y * RESTIR_RESERVOIR_BLOCK_SIZE
        + positionInBlock.x;
}

#endif // RESTIR_HELPERS_SLANG
