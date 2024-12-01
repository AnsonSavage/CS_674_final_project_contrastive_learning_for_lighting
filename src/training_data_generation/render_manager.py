import bpy

class RenderManager:
    def __init__(self, output_path, file_format='PNG', resolution_x=256, resolution_y=256, resolution_percentage=100):
        self.output_path = output_path
        bpy.context.scene.render.filepath = self.output_path
        bpy.context.scene.render.image_settings.file_format = file_format
        bpy.context.scene.render.resolution_x = resolution_x
        bpy.context.scene.render.resolution_y = resolution_y
        bpy.context.scene.render.resolution_percentage = resolution_percentage

    def render_image(self):
        bpy.ops.render.render(write_still=True)

# Example usage:
# render_manager = RenderManager('/path/to/output', file_format='JPEG', resolution_x=1280, resolution_y=720, resolution_percentage=50)
# render_manager.render_image()
