import bpy
import pathlib

class HDRIManager:
    """Responsible for managing the available HDRIs and setting them in the scene."""

    def __init__(self, hdri_directory: str, recursive: bool = True):
        self.available_hdris = self._get_available_hdris(hdri_directory, recursive)
    
    def _get_available_hdris(self, hdri_directory, recursive):
        """Returns a list of available HDRIs in the specified directory.
        
        :param hdri_directory: Path to the HDRI directory
        :param recursive: Whether to search subdirectories
        """
        hdri_directory = pathlib.Path(hdri_directory)
        if recursive:
            hdris = [hdri for hdri in hdri_directory.rglob('*') if hdri.suffix.lower() in [".exr", ".hdr", ".png", ".jpg"]]
        else:
            hdris = [hdri for hdri in hdri_directory.iterdir() if hdri.suffix.lower() in [".exr", ".hdr", ".png", ".jpg"]]
        return hdris


    def set_hdri(self, hdri_path: str, strength=1.0):
        """
        Set an HDRI environment texture in Blender's World settings.
        
        :param hdri_path: Full path to the HDRI image file
        """
        # Clear existing world nodes
        bpy.context.scene.world.node_tree.nodes.clear()
        
        # Create new World output node
        world_output = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        
        # Create Environment Texture node
        env_texture = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
        
        # Set the image for the Environment Texture
        env_texture.image = bpy.data.images.load(str(hdri_path))
        
        # Create Background node
        background = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeBackground')
        
        # Link nodes
        links = bpy.context.scene.world.node_tree.links
        links.new(env_texture.outputs["Color"], background.inputs["Color"])
        links.new(background.outputs["Background"], world_output.inputs["Surface"])
        
        # Optional: Set strength of the HDRI (default is 1.0)
        background.inputs["Strength"].default_value = strength