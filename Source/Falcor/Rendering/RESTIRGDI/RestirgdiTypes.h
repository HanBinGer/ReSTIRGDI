/***************************************************************************
 # Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RESTIR_TYPES_H
#define RESTIR_TYPES_H

#ifndef __cplusplus
#define uint32_t uint

#define RESTIR_TEX2D Texture2D
#define RESTIR_TEX2D_LOAD(t, pos, lod) t.Load(int3(pos, lod))
#define RESTIR_DEFAULT(value) = value

#endif // __cplusplus

#endif // RESTIR_TYPES_H
