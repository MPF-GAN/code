import bpy
import os

# Define input/output paths
input_filepath = os.path.abspath("./results/2.obj")
output_filepath = os.path.abspath("./results/2.obj")
pathD = os.path.abspath("./results/2.mtl")

# Clear the current scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import using the new 4.0+ OBJ importer
bpy.ops.wm.obj_import(filepath=input_filepath)

# Find the imported mesh object
imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
if not imported_objects:
    raise RuntimeError("No mesh object found in the imported file.")

obj = imported_objects[0]
bpy.context.view_layer.objects.active = obj

# Add Subdivision Surface modifier
subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 5
subsurf.subdivision_type = 'CATMULL_CLARK'
subsurf.use_creases = True

# Apply the modifier
bpy.ops.object.modifier_apply(modifier=subsurf.name)

# Export using the new 4.0+ OBJ exporter
bpy.ops.wm.obj_export(filepath=output_filepath, export_selected_objects=True)

if os.path.exists(pathD):
    os.remove(pathD)

print(f"Saved subdivided object to: {output_filepath}")
