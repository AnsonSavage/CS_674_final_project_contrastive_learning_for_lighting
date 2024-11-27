import bpy
import mathutils
import bmesh
import random

class CameraSpawner:
    def __init__(self, look_from_volume_name, look_at_volume_name, camera_name, seed):
        self.look_at_volume = bpy.data.objects.get(look_at_volume_name)
        self.look_from_volume = bpy.data.objects.get(look_from_volume_name)
        self.camera_name = camera_name
        self.seed = seed

    def update(self):
        self._set_volume_point_extraction_seeds(self.look_at_volume, "Look At Volume", self._get_next_seed())
        self._set_volume_point_extraction_seeds(self.look_at_volume, "Look At Volume", self._get_next_seed())

        depsgraph = bpy.context.evaluated_depsgraph_get()
        look_from_coords = self._get_vert_sequence_from_object(self.look_from_volume, depsgraph)
        look_at_coords = self._get_vert_sequence_from_object(self.look_at_volume, depsgraph)
        
        assert len(look_from_coords) == 1, f"Look from volume '{self.look_from_volume.name}' must have exactly one vertex."
        assert len(look_at_coords) == 1, f"Look at volume '{self.look_at_volume.name}' must have exactly one vertex."
        look_from = look_from_coords[0]
        look_at = look_at_coords[0]
        
        if look_at.z > look_from.z:
            z_diff = look_at.z - look_from.z
            look_at.z = look_from.z - z_diff
        
        camera = bpy.data.objects.get(self.camera_name)
        assert camera is not None, f"Camera '{self.camera_name}' not found in the scene."
        
        bpy.data.objects["look_from_visualizer"].location = look_from
        bpy.data.objects["look_at_visualizer"].location = look_at
        
        look_at_matrix = self.compute_look_at_matrix(look_from, look_at)
        
        camera.matrix_world = look_at_matrix
        
        print(f"Camera '{self.camera_name}' moved to coordinate: {look_from}")
        print(f"Camera '{self.camera_name}' now looking at: {look_at}")

    def compute_look_at_matrix(self, camera_position: mathutils.Vector, target_position: mathutils.Vector):
        """
        Computes a look-at transformation matrix for a camera.
        Args:
            camera_position (mathutils.Vector): The position of the camera.
            target_position (mathutils.Vector): The position the camera is looking at.
        Returns:
            mathutils.Matrix: The look-at transformation matrix.
        """
        camera_direction = (target_position - camera_position).normalized()

        up = mathutils.Vector((0, 0, 1))
        camera_right = camera_direction.cross(up).normalized()
        
        camera_up = camera_right.cross(camera_direction).normalized()

        rotation_transform = mathutils.Matrix([camera_right, camera_up, -camera_direction]).transposed().to_4x4()

        translation_transform = mathutils.Matrix.Translation(camera_position)
        print(translation_transform)
        look_at_transform = translation_transform @ rotation_transform
        return look_at_transform

    def _get_vert_sequence_from_object(self, obj, depsgraph):
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        verts = [v.co.copy() for v in bm.verts]
        bm.free()
        return verts
    
    def _set_volume_point_extraction_seeds(self, object, modifier_name, seed, socket_index=2):
        assert object is not None, f"Object '{object.name}' not found."
        object_name = object.name
    
        modifier = object.modifiers.get(modifier_name)
        assert modifier is not None, f"Modifier '{modifier_name}' not found on '{object_name}'."

        modifier[f"Socket_{socket_index}"] = seed
    
        object.modifiers.update()
        object.update_tag()
    
    def _get_next_seed(self):
        self.seed += 1
        return self.seed