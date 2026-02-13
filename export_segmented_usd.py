#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import json
import logging
import gzip
import io
import shutil
import tempfile
import zipfile
import numpy as np
import torch
import msgpack
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, Union, List

# PLY handling
from plyfile import PlyData, PlyElement

# USD imports
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol, UsdPhysics, Vt, UsdShade

# SciPy for Convex Hull
from scipy.spatial import ConvexHull

# Mesh Generation
import open3d as o3d
import tetgen
import trimesh
import meshio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. USD Conversion Utilities (Ported from ply_to_usdz_converter.py)
# ==============================================================================

@dataclass
class NamedSerialized:
    """Stores serialized data with a filename."""
    filename: str
    serialized: Union[str, bytes]

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        zip_file.writestr(self.filename, self.serialized)

@dataclass
class NamedUSDStage:
    """Stores a USD stage with a filename."""
    filename: str
    stage: Usd.Stage

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)

def create_nurec_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    density_activation: str = "sigmoid",
    scale_activation: str = "exp",
) -> Dict[str, Any]:
    """Create the NuRec template dictionary for USDZ export."""
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": density_activation,
                        "scale_activation": scale_activation,
                        "rotation_activation": "normalize",
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": 2,
                            "density_kernel_density_clamping": False,
                            "density_kernel_min_response": 0.0113,
                            "radiance_sph_degree": 3,
                        },
                        "transmittance_threshold": 0.001,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": False,
                    "projection": {
                        "n_rolling_shutter_iterations": 5,
                        "ut_dim": 3,
                        "ut_alpha": 1.0,
                        "ut_beta": 2.0,
                        "ut_kappa": 0.0,
                        "ut_require_all_sigma_points": False,
                        "image_margin_factor": 0.1,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": True,
                        "tight_opacity_bounding": True,
                        "tile_based": True,
                        "near_clip_distance": 0.2,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": 0},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
            },
        }
    }

    # Fill state dict with tensor data (converted to float16 for efficiency)
    dtype = np.float16
    state_dict = template["nre_data"]["state_dict"]

    state_dict[".gaussians_nodes.gaussians.positions"] = positions.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.rotations"] = rotations.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.scales"] = scales.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.densities"] = densities.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(dtype).tobytes()

    # Empty extra_signal tensor
    extra_signal = np.zeros((positions.shape[0], 0), dtype=dtype)
    state_dict[".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()

    # n_active_features as 64-bit integer
    state_dict[".gaussians_nodes.gaussians.n_active_features"] = np.array([n_active_features], dtype=np.int64).tobytes()

    # Store shapes
    state_dict[".gaussians_nodes.gaussians.positions.shape"] = list(positions.shape)
    state_dict[".gaussians_nodes.gaussians.rotations.shape"] = list(rotations.shape)
    state_dict[".gaussians_nodes.gaussians.scales.shape"] = list(scales.shape)
    state_dict[".gaussians_nodes.gaussians.densities.shape"] = list(densities.shape)
    state_dict[".gaussians_nodes.gaussians.features_albedo.shape"] = list(features_albedo.shape)
    state_dict[".gaussians_nodes.gaussians.features_specular.shape"] = list(features_specular.shape)
    state_dict[".gaussians_nodes.gaussians.extra_signal.shape"] = list(extra_signal.shape)
    state_dict[".gaussians_nodes.gaussians.n_active_features.shape"] = []

    return template

def initialize_usd_stage() -> Usd.Stage:
    """Initialize a new USD stage with standard settings."""
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")

    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage

def create_gauss_usd(model_filename: str, positions: np.ndarray) -> NamedUSDStage:
    """Create the USD stage containing the Gaussian volume."""
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)
    min_x, min_y, min_z = float(min_coord[0]), float(min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(max_coord[1]), float(max_coord[2])

    stage = initialize_usd_stage()

    # Render settings
    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    # Define UsdVol::Volume
    gauss_path = "/World/gauss"
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()

    # Default conversion matrix from 3DGRUT to USDZ coordinate system
    default_conv_tf = np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(*default_conv_tf.flatten()))

    # NuRec volume properties
    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define field assets
    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    # Set file paths for field assets
    nurec_relative_path = "./" + model_filename
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

    # Color correction matrix (identity)
    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([1, 0, 0, 0]))
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0, 1, 0, 0]))
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0, 0, 1, 0]))

    # Extent and crop boundaries
    gauss_prim.GetAttribute("extent").Set([[min_x, min_y, min_z], [max_x, max_y, max_z]])
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(0, 0, 0))
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(min_x, min_y, min_z))
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(max_x, max_y, max_z))

    gauss_prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)

def create_default_usd(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    """Create the default USD layer that references the gauss stage."""
    stage = initialize_usd_stage()
    
    # Use CoalescingDiagnosticDelegate to silence spurious warnings if needed, 
    # but in script it might be better to just let it run.
    
    prim = stage.OverridePrim(f"/World/{Path(gauss_stage.filename).stem}")
    prim.GetReferences().AddReference(gauss_stage.filename)

    # Copy render settings
    gauss_layer = gauss_stage.stage.GetRootLayer()
    if "renderSettings" in gauss_layer.customLayerData:
        new_settings = gauss_layer.customLayerData["renderSettings"]
        stage.SetMetadataByDictKey("customLayerData", "renderSettings", new_settings)

    return NamedUSDStage(filename="default.usda", stage=stage)

def write_usdz(output_path: Path, model_file: NamedSerialized, gauss_usd: NamedUSDStage, default_usd: NamedUSDStage):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        default_usd.save_to_zip(zip_file)
        model_file.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)

def convert_memory_to_usdz(
    gaussian_data: Dict[str, np.ndarray], 
    output_path: Path
):
    """Convert in-memory gaussian data to USDZ."""
    template = create_nurec_template(
        positions=gaussian_data["positions"],
        rotations=gaussian_data["rotations"],
        scales=gaussian_data["scales"],
        densities=gaussian_data["densities"],
        features_albedo=gaussian_data["features_albedo"],
        features_specular=gaussian_data["features_specular"],
        n_active_features=gaussian_data["n_active_features"],
    )

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        packed = msgpack.packb(template)
        f.write(packed)
    
    model_file = NamedSerialized(filename=output_path.stem + ".nurec", serialized=buffer.getvalue())
    gauss_usd = create_gauss_usd(model_file.filename, gaussian_data["positions"])
    default_usd = create_default_usd(gauss_usd)

    write_usdz(output_path, model_file, gauss_usd, default_usd)

# ==============================================================================
# 2. Segmentation & Parsing Utilities
# ==============================================================================

def load_ply_for_seg(path):
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    
    data = {}
    for prop in plydata.elements[0].properties:
        data[prop.name] = np.asarray(plydata.elements[0][prop.name])
    
    xyz = np.stack((data["x"], data["y"], data["z"]), axis=1)
    
    return xyz, data, plydata.elements[0]

def parse_gaussian_data(data_dict, max_sh_degree=3):
    """Parse raw PLY data dict into the structure needed for conversion."""
    num_gaussians = data_dict["x"].shape[0]
    
    positions = np.stack((data_dict["x"], data_dict["y"], data_dict["z"]), axis=1).astype(np.float32)
    densities = data_dict["opacity"][..., np.newaxis].astype(np.float32)
    
    # Albedo
    features_albedo = np.zeros((num_gaussians, 3), dtype=np.float32)
    features_albedo[:, 0] = data_dict["f_dc_0"]
    features_albedo[:, 1] = data_dict["f_dc_1"]
    features_albedo[:, 2] = data_dict["f_dc_2"]
    
    # Specular
    num_speculars = (max_sh_degree + 1) ** 2 - 1
    features_specular = np.zeros((num_gaussians, num_speculars * 3), dtype=np.float32)
    
    # Collect rest
    # We assume standard naming f_rest_0, f_rest_1...
    # The dictionary keys are available
    extra_f_names = [k for k in data_dict.keys() if k.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    
    expected_extra_f = 3 * num_speculars
    
    if len(extra_f_names) == expected_extra_f:
        temp = np.zeros((num_gaussians, expected_extra_f), dtype=np.float32)
        for idx, name in enumerate(extra_f_names):
            temp[:, idx] = data_dict[name]
        # Reshape: (N, 3*K) -> (N, 3, K) -> transpose -> (N, K*3)
        temp = temp.reshape((num_gaussians, 3, num_speculars))
        features_specular = temp.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))
    
    # Scales
    scale_names = [k for k in data_dict.keys() if k.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((num_gaussians, len(scale_names)), dtype=np.float32)
    for idx, name in enumerate(scale_names):
        scales[:, idx] = data_dict[name]
        
    # Rotations
    # Rotations
    rot_names = [k for k in data_dict.keys() if k.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rotations = np.zeros((num_gaussians, len(rot_names)), dtype=np.float32)
    for idx, name in enumerate(rot_names):
        rotations[:, idx] = data_dict[name]
        
    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "densities": densities,
        "features_albedo": features_albedo,
        "features_specular": features_specular,
        "n_active_features": max_sh_degree
    }

def compute_alignment(points):
    """
    Compute Centroid and Gravity-Aligned Rotation (Y-up) using PCA on XZ plane.
    Returns: (centroid, rotation_matrix_3x3)
    """
    if len(points) < 4:
        return np.mean(points, axis=0), np.eye(3)
        
    # 1. Centroid
    centroid = np.mean(points, axis=0) # (x, y, z)
    
    # 2. PCA on XZ plane for Rotation (Gravity Aligned)
    points_centered = points - centroid
    points_xz = points_centered[:, [0, 2]]
    
    try:
        # Covariance
        cov_xz = np.cov(points_xz, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_xz)
        
        # Sort eigenvectors (Largest first = Major Axis)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        
        # Construct 3D Rotation Matrix (Basis Vectors as Columns)
        # New X = Major Axis (U)
        # New Y = Up (0, 1, 0)
        # New Z = Minor Axis (V)
        
        # evecs columns are [U, V]
        u = evecs[:, 0]
        v = evecs[:, 1]
        
        # Ensure Right-Handed System (Y = Cross(Z, X) -> Z = Cross(X, Y)?)
        # Let's define X = (u[0], 0, u[1]), Y = (0, 1, 0), Z = (v[0], 0, v[1])
        # Check determinant or cross product
        # Cross((u0, 0, u1), (0, 1, 0)) = (-u1, 0, u0). 
        # v should be aligned with this.
        
        # Simply: Construct matrix and checking determinant is 1.
        R = np.eye(3)
        R[0, 0] = u[0]; R[0, 2] = u[1]
        R[1, 0] = 0.0;  R[1, 2] = 0.0
        R[2, 0] = v[0]; R[2, 2] = v[1]
        
        if np.linalg.det(R) < 0:
            # Flip Z to ensure right-handed
            R[:, 2] *= -1
            
        return centroid, R
        
    except Exception as e:
        print(f"Warning: Alignment failed: {e}. Using Identity.")
        return centroid, np.eye(3)

def create_tet_mesh(points: np.ndarray, output_path: str):
    """
    Generates a Tetrahedral Mesh from points using Open3D (Alpha Shape) -> TetGen.
    Saves as .vtk or similar format supported by Isaac Lab.
    """
    if len(points) < 4:
        return False
        
    try:
        # 1. Surface Reconstruction (Alpha Shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Alpha Shape - heuristic alpha based on point density
        # Compute average distance
        dists = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        alpha = 2.0 * avg_dist # heuristic
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        if len(mesh.vertices) < 4 or len(mesh.triangles) < 4:
            # Fallback to Convex Hull
            print("Alpha shape failed/too small, falling back to Convex Hull")
            mesh, _ = pcd.compute_convex_hull()
            
        mesh = mesh.compute_triangle_normals()
        
        # 2. Convert to Trimesh for TetGen
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        tm = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 3. Tetrahedralize
        # TetGen requires a watertight manifold, but we'll try our best.
        # "pq1.2" = Refine mesh (p) with quality (q) ratio 1.2
        try:
             # Using the python `tetgen` library
             tg = tetgen.TetGen(vertices, faces)
             # TetGen returns tuple: (nodes, elems, faces, edges, neighbors, poly_face_indices)
             result = tg.tetrahedralize(switches="pq1.2/10")
             nodes = result[0]
             elems = result[1]
        except Exception as e:
             print(f"TetGen failed: {e}. Trying convex hull fallback...")
             # Fallback to Convex Hull of points -> TetGen
             hull = trimesh.convex.convex_hull(points)
             tg = tetgen.TetGen(hull.vertices, hull.faces)
             result = tg.tetrahedralize(switches="pq1.2/10")
             nodes = result[0]
             elems = result[1]

        # 4. Save as VTK (using meshio)
        # TetGen returns nodes (vertices) and elems (tetrahedra)
        cells = [("tetra", elems)]
        
        meshio.write(
            output_path,
            meshio.Mesh(
                points=nodes,
                cells=cells
            )
        )
        print(f"Saved TetMesh to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating TetMesh: {e}")
        return False

# ==============================================================================
# 3. Main Workflow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export segmented USDZ scene with physics from SplatGraph")
    parser.add_argument("--ply_path", required=True, help="Input point cloud PLY")
    parser.add_argument("--graph_path", required=True, help="Input scene_graph.json")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for USD/USDZ files")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration for checkpoint loading")
    parser.add_argument("--threshold", type=float, default=0.6, help="Assignment threshold")
    parser.add_argument("--top_k", type=int, default=None, help="Export only the top K largest objects")
    parser.add_argument("--target_ids", type=str, default=None, help="Comma-separated list of object IDs to export")
    parser.add_argument("--obj_name_like", type=str, default=None, help="Comma-separated list of substrings to match against object names")
    parser.add_argument("--static_objects", action="store_true", help="Export objects as static colliders (disable RigidBodyAPI)")
    parser.add_argument("--skip_segmentation", action="store_true", help="Skip segmentation and export entire scene as single USDZ")
    parser.add_argument("--output_usd_name", type=str, default="scene.usd", help="Name of the master USD file")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- SKIP SEGMENTATION PATH ---
    if args.skip_segmentation:
        print("\n--- Exporting Full Scene (No Segmentation) ---\n")
        
        # 1. Load PLY
        xyz, ply_data, _ = load_ply_for_seg(args.ply_path)
        
        # 2. Convert to USDZ
        print(f"Converting full point cloud ({xyz.shape[0]} points) to USDZ...")
        parsed_data = parse_gaussian_data(ply_data)
        
        usdz_name = "gaussians_full.usdz"
        usdz_path = os.path.join(args.output_dir, usdz_name)
        convert_memory_to_usdz(parsed_data, Path(usdz_path))
        
        # 3. Create Master Stage
        master_usd_path = os.path.join(args.output_dir, args.output_usd_name)
        print(f"Creating master stage at {master_usd_path}...")
        stage = Usd.Stage.CreateNew(master_usd_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        world_prim = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_prim.GetPrim())
        
        # 4. Add Reference
        gauss_prim = UsdGeom.Xform.Define(stage, "/World/Gaussians")
        gauss_prim.GetPrim().GetReferences().AddReference(f"./{usdz_name}")
        
        # 5. Add Physics Scene (for ground plane compatibility etc)
        scene_prim = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene_prim.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene_prim.CreateGravityMagnitudeAttr(9.81)
        
        # 6. Add Ground Plane (Heuristic)
        min_z = np.min(xyz[:, 2])
        print(f"Adding Ground Plane at Z={min_z:.2f}")
        ground_path = "/World/GroundPlane"
        ground_xform = UsdGeom.Xform.Define(stage, ground_path)
        ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, float(min_z)))
        
        ground_mesh = UsdGeom.Cube.Define(stage, f"{ground_path}/CollisionBox")
        ground_mesh.CreateSizeAttr(1000.0) 
        ground_mesh.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.01))
        ground_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        ground_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())
        
        stage.GetRootLayer().Save()
        print("Done!")
        return

    # -------------------------------------------------------------
    # Step 1: Segmentation
    # -------------------------------------------------------------
    print("\n--- Step 1: Segmentation ---\n")
    
    # Load PLY
    xyz, ply_data, _ = load_ply_for_seg(args.ply_path)
    num_points = xyz.shape[0]
    
    # Load Checkpoint
    checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, tuple):
        model_params = checkpoint[0]
    else:
        model_params = checkpoint
        
    # Extract Logits/Codebooks
    if len(model_params) > 8:
        logits = model_params[7]
        codebooks = model_params[8]
    else:
        # Fallback logic or error
        print(f"Error: Model params length {len(model_params)} too short for LangSplat (expected > 8).")
        sys.exit(1)
    
    # Prepare Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if logits.ndim == 1:
        print(f"Error: Logits parameter is 1D (shape {logits.shape}). Expected 2D features.")
        sys.exit(1)

    logits = logits.to(device)
    codebooks = codebooks.to(device)
   
    if codebooks.ndim == 2:
        K, D = codebooks.shape
        L = 1
        codebooks = codebooks.unsqueeze(0)
    else:
        L, K, D = codebooks.shape
        
    # Compute Features
    weights_list = []
    for i in range(L):
        level_logits = logits[:, i*K : (i+1)*K]
        level_weights = torch.softmax(level_logits, dim=1)
        weights_list.append(level_weights)
    weights = torch.cat(weights_list, dim=1)
    codebooks_flat = codebooks.view(-1, D)
    
    gaussian_features = np.zeros((num_points, D), dtype=np.float32)
    chunk_size = 50000
    
    print("Computing features...")
    with torch.no_grad():
        for i in tqdm(range(0, num_points, chunk_size)):
            chunk_weights = weights[i:i+chunk_size]
            chunk_features = chunk_weights @ codebooks_flat
            chunk_features = chunk_features / (chunk_features.norm(dim=1, keepdim=True) + 1e-10)
            gaussian_features[i:i+chunk_size] = chunk_features.cpu().numpy()
            
    # Load Graph
    print(f"Loading graph {args.graph_path}...")
    with open(args.graph_path, 'r') as f:
        graph_data = json.load(f)
        
    obj_id_list = []
    obj_feat_list = []
    obj_centroid_list = []
    obj_names = []
    obj_metadata = []
    
    def collect_objs(obj_list):
        for obj in obj_list:
            feat = np.array(obj['feature'])
            feat = feat / (np.linalg.norm(feat) + 1e-10)
            centroid = obj.get('centroid', [0.0, 0.0, 0.0])
            
            obj_id_list.append(obj['id'])
            obj_feat_list.append(feat)
            obj_centroid_list.append(centroid)
            
            # Extract Name
            # Priority: metadata.name > physics.name > ID
            name = f"obj_{obj['id']}"
            if 'metadata' in obj and 'name' in obj['metadata']:
                name = obj['metadata']['name']
            elif 'physics' in obj and 'name' in obj['physics']:
                name = obj['physics']['name']
            
            obj_names.append(name)
            obj_metadata.append(obj) # Store full obj for composition later
            
            collect_objs(obj.get('children', []))
            
    collect_objs(graph_data.get("objects", []))
    num_objs = len(obj_id_list)
    print(f"Found {num_objs} objects in graph.")
    
    # Compute Assignment
    print("Assigning points...")
    
    # Check if point_indices are available (Semantic Clustering Mode)
    has_point_indices = False
    if len(obj_metadata) > 0 and 'point_indices' in obj_metadata[0]:
        has_point_indices = True
        print("Found precise 'point_indices' in scene graph. Using explicit Semantic Clustering assignment.")
    
    assignments = np.full(num_points, -1, dtype=np.int32)
    scores = np.zeros(num_points, dtype=np.float32)

    if has_point_indices:
        # 1. Precise Assignment from SAM Masks
        for idx, obj in enumerate(obj_metadata):
            if 'point_indices' in obj:
                indices = np.array(obj['point_indices'], dtype=np.int64)
                # Ensure within bounds
                valid_mask = indices < num_points
                valid_indices = indices[valid_mask]
                
                assignments[valid_indices] = idx
                scores[valid_indices] = 1.0 # Max score for explicit assignment
        
        # Calculate Unassigned
        unassigned_count = (assignments == -1).sum()
        print(f"Explicit assignment complete. {unassigned_count} points remain unassigned (background).")

    else:
        # 2. Heuristic Assignment (Legacy / Fallback)
        print("No 'point_indices' found. Using Feature Similarity + Spatial Heuristic.")
        g_feats_th = torch.from_numpy(gaussian_features).float().to(device)
        o_feats_th = torch.from_numpy(np.stack(obj_feat_list)).float().to(device)
        g_xyz_th = torch.from_numpy(xyz).float().to(device)
        o_centroids_th = torch.from_numpy(np.array(obj_centroid_list)).float().to(device)
        
        spatial_weight = 0.8 # Slightly reduce from 1.0 to allow more feature influence
        
        with torch.no_grad():
            for i in tqdm(range(0, num_points, chunk_size)):
                g_chunk = g_feats_th[i:i+chunk_size]
                g_xyz_chunk = g_xyz_th[i:i+chunk_size]
                
                sim = g_chunk @ o_feats_th.T
                dist = torch.cdist(g_xyz_chunk, o_centroids_th)
                final_score = sim - (dist * spatial_weight)
                
                best_score, best_idx = torch.max(final_score, dim=1)
                best_raw_sim = torch.gather(sim, 1, best_idx.unsqueeze(1)).squeeze(1)
                
                assignments[i:i+chunk_size] = best_idx.cpu().numpy()
                scores[i:i+chunk_size] = best_raw_sim.cpu().numpy()

    # -------------------------------------------------------------
    # Step 2: Generation & Composition
    # -------------------------------------------------------------
    print("\n--- Step 2: Generation & Composition ---\n")
    
    # Init Master Stage
    master_usd_path = os.path.join(args.output_dir, args.output_usd_name)
    stage = Usd.Stage.CreateNew(master_usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    world_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world_prim.GetPrim())
    
    # Helper to add object to stage
    def add_object_to_stage(name, oid, usdz_rel_path, obj_meta=None, points_canonical=None, centroid=None, rotation=None):
        # Sanitize name
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        prim_path = f"/World/{safe_name}_{oid}"
        
        rb_xform = UsdGeom.Xform.Define(stage, prim_path)
        rb_prim = rb_xform.GetPrim()
        
        # === 1. TRANSFORMS ===
        # Apply Centroid Translation
        if centroid is not None:
            rb_xform.AddTranslateOp().Set(Gf.Vec3d(float(centroid[0]), float(centroid[1]), float(centroid[2])))
            
        # Apply Alignment Rotation (Parent is Aligned)
        if rotation is not None:
            # Convert 3x3 numpy to 4x4 matrix
            mat = np.eye(4)
            mat[:3, :3] = rotation
            # USD Transpose? NumPy is Row-Major? Gf is Row-Major?
            # GfMatrix4d constructor takes arguments row by row.
            # But GfMatrix4d is usually used as M * v (column vectors) in math, but stored?
            # Let's trust standard flattening.
            # Cast to standard floats for Boost.Python
            flat_mat = [float(x) for x in mat.flatten()]
            rb_xform.AddTransformOp().Set(Gf.Matrix4d(*flat_mat)) # Rotate Parent
            
        # Physics (Applied to Parent)
        if obj_meta:
            # Check for Soft Body
            # Priority 1: LLM Prediction in 'usd_physics'
            is_soft = False
            usd_phys = obj_meta.get("usd_physics", {})
            legacy_phys = obj_meta.get("physics", {})
            
            if "deformable" in usd_phys and "enabled" in usd_phys["deformable"]:
                is_soft = usd_phys["deformable"]["enabled"]
                print(f"  -> Object {name}: Using LLM predicted Deformable={is_soft}")
            else:
                # Priority 2: Keyword Fallback (Legacy)
                soft_keywords = ["fabric", "plush", "rubber", "sponge", "cloth", "pillow", "blanket", "soft", "pouch", "gummy"]
                
                # Check name, material, description
                meta_str = ""
                if "metadata" in obj_meta:
                    m = obj_meta["metadata"]
                    meta_str = f"{m.get('name','')} {m.get('material','')} {m.get('description','')} {m.get('motion_type','')}".lower()
                elif "physics" in obj_meta:
                    m = obj_meta["physics"]
                    meta_str = f"{m.get('name','')} {m.get('material','')} {m.get('motion_type','')}".lower()
                
                if any(k in meta_str for k in soft_keywords):
                    is_soft = True
                    print(f"  -> Object {name}: Fallback Soft Body detection (matched keywords)")

            # Mass
            mass_val = 1.0
            if "mass" in usd_phys and "mass" in usd_phys["mass"]:
                mass_val = usd_phys["mass"]["mass"]
            elif "mass_kg" in legacy_phys:
                mass_val = legacy_phys["mass_kg"]
                
            mass_api = UsdPhysics.MassAPI.Apply(rb_prim)
            mass_api.CreateMassAttr(mass_val)
            
            # BRANCH: Soft vs Rigid vs Static
            if is_soft or args.static_objects:
                if args.static_objects:
                     print(f"  -> Object {name}: Static Rigid Body (RigidBodyAPI Disabled)")
                # Soft Body or Static: Do NOT apply RigidBodyAPI.
                
                if is_soft:
                     # Export Tetrahedral Mesh for Isaac Lab to load
                     # We need the points relative to the CENTROID because the object Xform applies the centroid translation.
                     # BUT wait: The 'points_canonical' argument should contain points centered at (0,0,0) if centroid was subtracted.
                     # Let's verify what points_canonical contains.
                     
                     # Check if points_canonical is provided
                     if points_canonical is not None:
                         # It is just points - centroid.
                         # But tetgen needs to generate the mesh in this LOCAL frame.
                         vtk_filename = f"{safe_name}_{oid}_physics.vtk"
                         vtk_path = os.path.join(args.output_dir, vtk_filename)
                         
                         success = create_tet_mesh(points_canonical, vtk_path)
                         if success:
                             print(f"  -> Object {name}: Generated Deformable Mesh {vtk_filename}")
                             # We DON'T add PhysxDeformableBodyAPI here because we will load it via Isaac Lab's DeformableObjectCfg
                             # which takes the .vtk file directly.
                             # However, we MIGHT want to add a reference or metadata?
                             # For now, we assume the loading script will look for `_physics.vtk`
                         else:
                             print(f"  -> Object {name}: Deformable Mesh generation failed.")
                
                pass
            else:
                # Rigid Body
                phys_api = UsdPhysics.RigidBodyAPI.Apply(rb_prim)
                phys_api.CreateRigidBodyEnabledAttr(True)
                
                # Rigid Collision API on the Xform
                # Only needed for dynamic bodies usually (as a root for compound shapes)
                coll_api = UsdPhysics.CollisionAPI.Apply(rb_prim)
                coll_api.CreateCollisionEnabledAttr(True)
            
            # GENERATE COLLISION MESH
            # Priority 1: Use Generative Fill Mesh (if available)
            if "collision_mesh" in usd_phys and os.path.exists(usd_phys["collision_mesh"]):
                try:
                    import trimesh
                    mesh_path = usd_phys["collision_mesh"]
                    print(f"  -> Object {name}: Using Generative Fill Mesh: {mesh_path}")
                    
                    filled_mesh = trimesh.load(mesh_path)
                    
                    # Create Collision Mesh Prim
                    coll_mesh_path = f"{prim_path}/CollisionMesh"
                    coll_mesh = UsdGeom.Mesh.Define(stage, coll_mesh_path)
                    
                    # Vertices
                    coll_mesh.CreatePointsAttr(filled_mesh.vertices.tolist())
                    coll_mesh.CreateFaceVertexCountsAttr([len(f) for f in filled_mesh.faces])
                    coll_mesh.CreateFaceVertexIndicesAttr(filled_mesh.faces.flatten().tolist())
                    
                    # Set invisible (Proxy)
                    coll_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
                    coll_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                    
                    # Apply Physics Schema (Soft/Rigid)
                    # ... [Use same logic as below] ...
                    prim = coll_mesh.GetPrim()
                    
                    if is_soft:
                        # Soft Body Setup
                        prim.CreateAttribute("physxDeformable:deformableEnabled", Sdf.ValueTypeNames.Bool).Set(True)
                        prim.CreateAttribute("physxDeformable:selfCollision", Sdf.ValueTypeNames.Bool).Set(True)
                        prim.CreateAttribute("physxDeformable:solverPositionIterationCount", Sdf.ValueTypeNames.UInt).Set(4)
                        
                        # Material binding
                        mat_path = f"{prim_path}/PhysicsMaterial"
                        mat = UsdShade.Material.Define(stage, mat_path)
                        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                        
                        roughness = 0.5
                        if "metadata" in obj_meta and "roughness" in obj_meta["metadata"]:
                            roughness = float(obj_meta["metadata"]["roughness"])
                        elif "deformable" in usd_phys and "friction" in usd_phys["deformable"]:
                             roughness = float(usd_phys["deformable"]["friction"])
                        
                        friction = 0.1 + (roughness * 0.9)
                        phys_mat.CreateDynamicFrictionAttr(friction)
                        phys_mat.CreateStaticFrictionAttr(friction)
                        phys_mat.CreateRestitutionAttr(0.0)
                        
                        UsdShade.MaterialBindingAPI(prim).Bind(mat)
                        UsdPhysics.CollisionAPI.Apply(prim)
                        
                    else:
                        # Rigid Body Setup
                        mat_path = f"{prim_path}/PhysicsMaterial"
                        mat = UsdShade.Material.Define(stage, mat_path)
                        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                        
                        roughness = 0.5
                        if "metadata" in obj_meta and "roughness" in obj_meta["metadata"]:
                             roughness = float(obj_meta["metadata"]["roughness"])
                        
                        friction = 0.1 + (roughness * 0.9)
                        phys_mat.CreateDynamicFrictionAttr(friction)
                        phys_mat.CreateStaticFrictionAttr(friction)
                        
                        UsdShade.MaterialBindingAPI(prim).Bind(mat)
                        
                        # Apply TriangleMesh Collision (since it's a generic mesh, likely concave?)
                        # If Generative Fill produces a closed mesh, Convex Decomposition might be better?
                        # But for now let's use "none" approximation which implies Triangle Mesh (if supported) or ConvexHull
                        # Actually, better to explicitly set approximation to "convexHull" if it's convex, or "convexDecomposition" if not.
                        # Generative Fill usually produces complex shapes.
                        mesh_coll_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mesh_coll_api.CreateApproximationAttr("convexDecomposition")
                        
                        UsdPhysics.CollisionAPI.Apply(prim)

                except Exception as e:
                    print(f"Warning: Failed to load/use filled mesh {usd_phys['collision_mesh']}: {e}. Fallback to Hull.")
                    # Pass through to hull logic
                    pass

            # Priority 2: Convex Hull (Fallback)
            if (not "collision_mesh" in usd_phys or not os.path.exists(str(usd_phys.get("collision_mesh", "")))) and points_canonical is not None and len(points_canonical) >= 4:
                try:
                    hull = ConvexHull(points_canonical)
                    
                    # Create Collision Mesh Prim
                    coll_mesh_path = f"{prim_path}/CollisionMesh"
                    coll_mesh = UsdGeom.Mesh.Define(stage, coll_mesh_path)
                    
                    # Vertices
                    hull_verts_idx = hull.vertices
                    hull_verts = points_canonical[hull_verts_idx]
                    old_to_new = {old: new for new, old in enumerate(hull_verts_idx)}
                    
                    # Faces
                    face_vertex_counts = []
                    face_vertex_indices = []
                    for simplex in hull.simplices:
                        face_vertex_counts.append(len(simplex))
                        for v_idx in simplex:
                            face_vertex_indices.append(old_to_new[v_idx])
                            
                    coll_mesh.CreatePointsAttr(hull_verts)
                    coll_mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
                    coll_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
                    
                    # Set invisible (Proxy)
                    coll_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
                    coll_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                    
                    if is_soft:
                        # === SOFT BODY SETUP (Manual Schema Baking) ===
                        # We verified this works for Isaac Sim to trigger parsing.
                        prim = coll_mesh.GetPrim()
                        
                        # 1. Enable Deformable
                        prim.CreateAttribute("physxDeformable:deformableEnabled", Sdf.ValueTypeNames.Bool).Set(True)
                        
                        # 2. Solver Configuration
                        prim.CreateAttribute("physxDeformable:selfCollision", Sdf.ValueTypeNames.Bool).Set(True)
                        prim.CreateAttribute("physxDeformable:solverPositionIterationCount", Sdf.ValueTypeNames.UInt).Set(4)

                        # 4. Friction (Crucial for Manipulation)
                        # Create a Physics Material
                        mat_path = f"{prim_path}/PhysicsMaterial"
                        mat = UsdShade.Material.Define(stage, mat_path)
                        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                        
                        # Get Roughness from Metadata
                        roughness = 0.5
                        if "metadata" in obj_meta and "roughness" in obj_meta["metadata"]:
                            roughness = float(obj_meta["metadata"]["roughness"])
                        elif "deformable" in usd_phys and "friction" in usd_phys["deformable"]: # Future proofing
                            roughness = float(usd_phys["deformable"]["friction"])
                            
                        # Map Roughness -> Friction (0.0->0.1, 1.0->1.0)
                        friction = 0.1 + (roughness * 0.9)
                        
                        phys_mat.CreateDynamicFrictionAttr(friction)
                        phys_mat.CreateStaticFrictionAttr(friction)
                        phys_mat.CreateRestitutionAttr(0.0) # No bounce for fabrics
                        
                        # Bind Material to Mesh
                        UsdShade.MaterialBindingAPI(prim).Bind(mat)
                        
                        print(f"  -> Object {name}: Bound Physics Material with Friction {friction:.2f}")

                        # Apply CollisionAPI
                        UsdPhysics.CollisionAPI.Apply(prim)
                        
                    else:
                        # === RIGID BODY SETUP ===
                        # Also apply friction for rigid bodies!
                        
                        # Create Physics Material
                        mat_path = f"{prim_path}/PhysicsMaterial"
                        mat = UsdShade.Material.Define(stage, mat_path)
                        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                        
                        # Get Roughness
                        roughness = 0.5
                        if "metadata" in obj_meta and "roughness" in obj_meta["metadata"]:
                            roughness = float(obj_meta["metadata"]["roughness"])
                            
                        friction = 0.1 + (roughness * 0.9)
                        phys_mat.CreateDynamicFrictionAttr(friction)
                        phys_mat.CreateStaticFrictionAttr(friction)
                        
                        UsdShade.MaterialBindingAPI(coll_mesh.GetPrim()).Bind(mat)
                        
                        # Apply MeshCollisionAPI
                        mesh_coll_api = UsdPhysics.MeshCollisionAPI.Apply(coll_mesh.GetPrim())
                        mesh_coll_api.CreateApproximationAttr("convexHull")
                        
                        # Apply CollisionAPI
                        UsdPhysics.CollisionAPI.Apply(coll_mesh.GetPrim())

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Warning: Failed to generate convex hull for {name}: {e}")
        
        # Visual
        vis_xform = UsdGeom.Xform.Define(stage, f"{prim_path}/Visual")
        
        # Counter-Rotate Visuals (if Parent is Rotated)
        if rotation is not None:
             # Inverse Rotation = Transpose for Orthogonal Matrix
            rot_inv = rotation.T
            mat_inv = np.eye(4)
            mat_inv[:3, :3] = rot_inv
            flat_inv = [float(x) for x in mat_inv.flatten()]
            vis_xform.AddTransformOp().Set(Gf.Matrix4d(*flat_inv))
            
        vis_xform.GetPrim().GetReferences().AddReference(usdz_rel_path)

    # Determine Active Objects
    active_indices = set(range(num_objs))
    
    # 1. Filter by target_ids
    if args.target_ids:
        target_id_set = set(int(x) for x in args.target_ids.split(","))
        active_indices = {i for i in active_indices if obj_id_list[i] in target_id_set}
        print(f"Filtered to {len(active_indices)} objects by target_ids.")
    
    # NEW: Filter by name substring
    elif args.obj_name_like:
        substrings = [s.strip().lower() for s in args.obj_name_like.split(",")]
        matched_indices = set()
        for i in active_indices:
            # Check if any substring is in the object name (case-insensitive)
            name_lower = obj_names[i].lower()
            if any(sub in name_lower for sub in substrings):
                matched_indices.add(i)
        active_indices = matched_indices
        print(f"Filtered to {len(active_indices)} objects matching names '{args.obj_name_like}'.")
        
    # 2. Filter by top_k (if still applicable)
    elif args.top_k is not None:
        # We need object sizes. This requires a quick pass or updated Assignment logic.
        # assignments has shape (N,).
        # Count occurrences.
        print("Counting object points for top_k filtering...")
        # Only count points that meet threshold
        valid_mask = scores >= args.threshold
        valid_assignments = assignments[valid_mask]
        
        counts = np.bincount(valid_assignments, minlength=num_objs)
        # Get indices of top k
        # argsort is ascending, so take last k
        top_k_indices = np.argsort(counts)[-args.top_k:]
        active_indices = set(top_k_indices)
        print(f"Filtered to top {args.top_k} objects by size.")

    # 1. Background
    # Background includes points with low score OR points belonging to inactive objects
    
    # Create mask for points assigned to INACTIVE objects
    # We can do this efficiently
    # active_mask[i] is True if object i is active
    is_obj_active = np.zeros(num_objs, dtype=bool)
    is_obj_active[list(active_indices)] = True
    
    # For each point, check if its assigned object is active
    # assignments might have -1, handle carefully
    # If assignment is -1, it's not active anyway.
    
    # Safe lookup: map -1 to False (index -1 in numpy usually wraps, but we can clamp or use logic)
    # Actually simpler: 
    # point_is_active_obj = is_obj_active[assignments] (if assignments are valid indices)
    # assignments has -1. 
    safes_assignments = assignments.copy()
    safes_assignments[safes_assignments == -1] = 0 # Dummy valid
    point_assigned_active = is_obj_active[safes_assignments]
    point_assigned_active[assignments == -1] = False
    
    # Final Background Mask:
    # (Score < Threshold) OR (Assigned Object is NOT Active)
    bg_mask = (scores < args.threshold) | (~point_assigned_active)
    
    bg_indices = np.where(bg_mask)[0]
    if len(bg_indices) > 0:
        print(f"Processing Background ({len(bg_indices)} points)...")
        bg_data = {k: v[bg_indices] for k, v in ply_data.items()}
        
        bg_usdz_name = "background.usdz"
        bg_usdz_path = os.path.join(args.output_dir, bg_usdz_name)
        
        # Convert
        parsed_bg = parse_gaussian_data(bg_data)
        convert_memory_to_usdz(parsed_bg, Path(bg_usdz_path))
        
        # Add to Stage (Static)
        bg_prim = UsdGeom.Xform.Define(stage, "/World/Background")
        bg_prim.GetPrim().GetReferences().AddReference(f"./{bg_usdz_name}")
        
    # 2. Objects
    # Only iterate active indices
    for idx in sorted(list(active_indices)):
        oid = obj_id_list[idx]
        name = obj_names[idx]
        obj_meta = obj_metadata[idx]
        
        obj_mask = (assignments == idx) & (scores >= args.threshold)
        obj_indices = np.where(obj_mask)[0]
        
        if len(obj_indices) == 0:
            continue
            
        print(f"Processing Object {name} ({len(obj_indices)} points)...")
        
        # Extract Data
        obj_data = {k: v[obj_indices] for k, v in ply_data.items()}
        
        # --- ALIGNMENT & CENTERING ---
        # 1. Stack points
        obj_pts_raw = np.stack((obj_data["x"], obj_data["y"], obj_data["z"]), axis=1)
        
        # 2. Compute Alignment (World -> Body)
        # centroid: World position of pivot
        # R_wb: Rotation matrix that projects World -> Canonical (Body)
        centroid, R_wb = compute_alignment(obj_pts_raw)
        
        # 3. Center Points (Translation only)
        # We must NOT rotate the points here because SH coefficients 
        # are tied to the global frame orientation.
        obj_pts_centered = obj_pts_raw - centroid
        
        # Update obj_data for USDZ export
        obj_data["x"] = obj_pts_centered[:, 0]
        obj_data["y"] = obj_pts_centered[:, 1]
        obj_data["z"] = obj_pts_centered[:, 2]
        
        # 4. Compute Canonical Points for Concave Hull (Physics)
        # Canonical = Centered @ R_wb.T (Project onto PCA axes)
        # R_wb rows are eigenvectors U, V. R_wb.T columns are U, V.
        # v @ Matrix_Col_U = dot(v, U). Correct.
        obj_pts_canonical = obj_pts_centered @ R_wb.T

        # Save USDZ
        # Clean name for filename
        clean_name = name.replace(" ", "_").replace("/", "-")
        usdz_name = f"{clean_name}_{oid}.usdz"
        usdz_path = os.path.join(args.output_dir, usdz_name)
        
        parsed_obj = parse_gaussian_data(obj_data)
        convert_memory_to_usdz(parsed_obj, Path(usdz_path))
        
        # Add to Master Stage
        # rotation arg: Body -> World transform (R_wb.T)
        obj_rotation = R_wb.T # Renamed from R_bw
        
        add_object_to_stage(
            name=name,
            oid=oid,
            usdz_rel_path=f"./{usdz_name}",
            obj_meta=obj_meta,
            points_canonical=obj_pts_canonical, # Passed for TetGen
            centroid=centroid,
            rotation=obj_rotation
        )

    # 3. Add Ground Plane
    if num_objs > 0:
        # Heuristic: Find min Z of background or scene
        min_z = np.min(xyz[:, 2])
        print(f"Adding Ground Plane at Z={min_z:.2f}")
        
        ground_path = "/World/GroundPlane"
        ground_xform = UsdGeom.Xform.Define(stage, ground_path)
        ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, float(min_z)))
        
        # Create a large box as ground (Plane sometimes behaves weirdly in pure USD without PhysicsScene def)
        # Actually UsdPhysics often requires a PhysicsScene defined at root.
        
        # Define PhysicsScene
        scene_prim = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene_prim.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene_prim.CreateGravityMagnitudeAttr(9.81)
        
        # Ground Box (Invisible)
        ground_mesh = UsdGeom.Cube.Define(stage, f"{ground_path}/CollisionBox")
        ground_mesh.CreateSizeAttr(1000.0) # Large
        ground_mesh.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.01)) # Flatten
        ground_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        ground_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        
        # Collision for Ground
        UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())
        # RigidBody? No, static kinematic by default if no RigidBodyAPI.
        # But we need CollisionAPI.
        
    # Save Master
    print(f"Saving master stage to {master_usd_path}...")
    stage.GetRootLayer().Save()
    print("Done!")

if __name__ == "__main__":
    main()
