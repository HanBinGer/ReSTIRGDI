/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import LightSample;
import Utils.Debug.PixelDebug;
import Params;

/** Represents a reservoir containing a weighted light sample obtained through resampled importance sampling.
*/
struct Reservoir
{
    LightSample lightSample;    ///< Selected light sample.
    uint M;                     ///< Number of samples seen so far.
    float weight;               ///< Probabilistic weight.
    uint pathSample;
    float targetPdf;

    static Reservoir createEmpty( uint pathSample = 2)
    {
        return { LightSample::createInvalid(), 0.f, 0.f, pathSample, 0.f };
    }

    bool isLightSampleValid()
    {
        return lightSample.isValid();
    }

    bool isReservoirValid()
    {
        return isLightSampleValid();
    }

    void printReservoir()
    {
        print("Reservoir.lightSample = ", lightSample.getIndex());
        print("Reservoir.M = ", M);
        print("Reservoir.weight = ", weight);
        print("Reservoir.pathSample = ", pathSample);
        print("Reservoir.targetPdf = ", targetPdf);
    }
};
